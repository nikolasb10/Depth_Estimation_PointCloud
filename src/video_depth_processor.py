"""
Video Depth Processing Module

Processes videos frame-by-frame for depth estimation and 3D reconstruction.
Implements STEP 1 & STEP 2 from the 3D reconstruction pipeline:
- Depth → 3D Point Cloud conversion with camera intrinsics
- Video → Dense 3D Scene reconstruction with motion estimation
"""

import cv2
import numpy as np
import open3d as o3d
from typing import Tuple, List, Optional, Dict, Any
import logging
from pathlib import Path
import tempfile
import os
from PIL import Image

from depth_estimator import DepthEstimator
from pointcloud_generator import PointCloudGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_VIDEO = "sample_videos/sequence_00.mp4"

class CameraIntrinsics:
    """Camera intrinsics parameters for 3D projection."""
    
    def __init__(self, fx: float, fy: float, cx: float, cy: float, 
                 width: int, height: int):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        
        # Create intrinsic matrix
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    @classmethod
    def from_fov(cls, width: int, height: int, fov_degrees: float = 60.0):
        """Create camera intrinsics from field of view."""
        fov_rad = np.radians(fov_degrees)
        fx = fy = width / (2.0 * np.tan(fov_rad / 2.0))
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx, fy, cx, cy, width, height)
    
    @classmethod
    def default_webcam(cls, width: int = 640, height: int = 480):
        """Default webcam intrinsics."""
        fx = fy = 500.0  # Typical webcam focal length
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx, fy, cx, cy, width, height)




class MotionEstimator:
    """Estimates camera motion between frames using feature matching."""
    
    def __init__(self, detector_type: str = "ORB"):
        """
        Initialize motion estimator.
        
        Args:
            detector_type: Feature detector type ("ORB", "SIFT", "AKAZE")
        """
        self.detector_type = detector_type
        
        if detector_type == "ORB":
            self.detector = cv2.ORB_create(nfeatures=1000)
        elif detector_type == "SIFT":
            self.detector = cv2.SIFT_create()
        elif detector_type == "AKAZE":
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
        
        # Feature matcher
        if detector_type == "ORB":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    def estimate_motion(self, img1: np.ndarray, img2: np.ndarray, 
                       camera_intrinsics: CameraIntrinsics) -> Tuple[np.ndarray, bool]:
        """
        Estimate camera motion between two frames.
        
        Args:
            img1: First image (grayscale)
            img2: Second image (grayscale)
            camera_intrinsics: Camera parameters
            
        Returns:
            (transformation_matrix, success): 4x4 transformation matrix and success flag
        """
        # Detect keypoints and descriptors
        kp1, desc1 = self.detector.detectAndCompute(img1, None)
        kp2, desc2 = self.detector.detectAndCompute(img2, None)
        
        if desc1 is None or desc2 is None or len(desc1) < 8 or len(desc2) < 8:
            logger.warning("Insufficient features detected")
            return np.eye(4), False
        
        # Match features
        matches = self.matcher.match(desc1, desc2)
        
        if len(matches) < 8:
            logger.warning("Insufficient matches found")
            return np.eye(4), False
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Find essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, camera_intrinsics.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        if E is None:
            logger.warning("Failed to compute essential matrix")
            return np.eye(4), False
        
        # Recover pose
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, camera_intrinsics.K, mask=mask)
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        return T, True


class VideoDepthProcessor:
    """Main class for processing videos with depth estimation and 3D reconstruction."""
    
    def __init__(self, model_name: str = "Intel/dpt-hybrid-midas"):
        """
        Initialize video depth processor.
        
        Args:
            model_name: Depth estimation model to use
        """
        self.depth_estimator = DepthEstimator(model_name=model_name)
        self.motion_estimator = MotionEstimator()
        
        # Processing parameters
        self.max_depth = 10.0
        self.frame_skip = 1  # Process every N frames
        self.max_frames = 100  # Maximum frames to process
        
    def process_video_to_pointclouds(self, video_path: str, 
                                   camera_intrinsics: Optional[CameraIntrinsics] = None,
                                   output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process video frame-by-frame to generate point clouds.
        
        Args:
            video_path: Path to input video
            camera_intrinsics: Camera parameters (estimated if None)
            output_dir: Directory to save point clouds (optional)
            
        Returns:
            List of dictionaries containing frame data
        """
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
        
        # Estimate camera intrinsics if not provided
        if camera_intrinsics is None:
            camera_intrinsics = CameraIntrinsics.default_webcam(width, height)
            logger.info("Using default camera intrinsics")
        
        # Initialize point cloud generator with camera intrinsics
        pcg = PointCloudGenerator()
        pcg.set_camera_intrinsics(
            camera_intrinsics.fx, camera_intrinsics.fy,
            camera_intrinsics.cx, camera_intrinsics.cy,
            camera_intrinsics.width, camera_intrinsics.height
        )
        
        # Process frames
        results = []
        frame_idx = 0
        prev_gray = None
        cumulative_pose = np.eye(4)
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        while cap.isOpened() and frame_idx < min(frame_count, self.max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_idx % self.frame_skip != 0:
                frame_idx += 1
                continue
            
            logger.info(f"Processing frame {frame_idx}/{min(frame_count, self.max_frames)}")
            
            # Convert to RGB for depth estimation
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Estimate depth
            try:
                depth_map, _ = self.depth_estimator.estimate_depth_from_array(rgb_frame)
            except Exception as e:
                logger.error(f"Depth estimation failed for frame {frame_idx}: {e}")
                frame_idx += 1
                continue
            
            # Estimate motion (if not first frame)
            relative_pose = np.eye(4)
            if prev_gray is not None:
                relative_pose, motion_success = self.motion_estimator.estimate_motion(
                    prev_gray, gray_frame, camera_intrinsics
                )
                if motion_success:
                    cumulative_pose = cumulative_pose @ relative_pose
            
            # Convert depth to point cloud
            try:
                pointcloud = pcg.depth_to_pointcloud(rgb_frame, depth_map, self.max_depth)
                
                # Transform point cloud by camera pose
                if not np.allclose(cumulative_pose, np.eye(4)):
                    pointcloud.transform(cumulative_pose)
                
            except Exception as e:
                logger.error(f"Point cloud generation failed for frame {frame_idx}: {e}")
                frame_idx += 1
                continue
            
            # Save results
            frame_data = {
                'frame_idx': frame_idx,
                'rgb_frame': rgb_frame,
                'depth_map': depth_map,
                'pointcloud': pointcloud,
                'camera_pose': cumulative_pose.copy(),
                'relative_pose': relative_pose
            }
            
            # Save point cloud to file if output directory specified
            if output_dir:
                ply_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.ply")
                o3d.io.write_point_cloud(ply_path, pointcloud)
                frame_data['ply_path'] = ply_path
            
            results.append(frame_data)
            prev_gray = gray_frame
            frame_idx += 1
        
        cap.release()
        logger.info(f"Processed {len(results)} frames")
        
        return results
    
    def merge_pointclouds(self, frame_results: List[Dict[str, Any]], 
                         voxel_size: float = 0.05) -> o3d.geometry.PointCloud:
        """
        Merge point clouds from multiple frames into a single dense reconstruction.
        
        Args:
            frame_results: Results from process_video_to_pointclouds
            voxel_size: Voxel size for downsampling
            
        Returns:
            Merged point cloud
        """
        logger.info(f"Merging {len(frame_results)} point clouds")
        
        if not frame_results:
            return o3d.geometry.PointCloud()
        
        # Start with first point cloud
        merged_pcd = frame_results[0]['pointcloud']
        
        # Add subsequent point clouds
        for i, frame_data in enumerate(frame_results[1:], 1):
            logger.info(f"Merging point cloud {i}/{len(frame_results)-1}")
            
            pcd = frame_data['pointcloud']
            merged_pcd += pcd
        
        # Use PointCloudGenerator for processing
        pcg = PointCloudGenerator()
        
        # Downsample to reduce redundancy
        if voxel_size > 0:
            logger.info(f"Downsampling with voxel size {voxel_size}")
            merged_pcd = pcg.downsample_pointcloud(merged_pcd, voxel_size)
        
        # Remove statistical outliers
        merged_pcd = pcg.filter_pointcloud(merged_pcd)
        
        logger.info(f"Final merged point cloud has {len(merged_pcd.points)} points")
        
        return merged_pcd
    
    def process_video_to_scene(self, video_path: str, 
                              camera_intrinsics: Optional[CameraIntrinsics] = None,
                              output_path: Optional[str] = None) -> o3d.geometry.PointCloud:
        """
        Complete pipeline: video → dense 3D scene reconstruction.
        
        Args:
            video_path: Path to input video
            camera_intrinsics: Camera parameters
            output_path: Path to save final point cloud
            
        Returns:
            Merged 3D scene point cloud
        """
        # Process video frame by frame
        frame_results = self.process_video_to_pointclouds(video_path, camera_intrinsics)
        
        # Merge into single scene
        scene_pcd = self.merge_pointclouds(frame_results)
        
        # Save if output path specified
        if output_path:
            o3d.io.write_point_cloud(output_path, scene_pcd)
            logger.info(f"Saved merged scene to: {output_path}")
        
        return scene_pcd


if __name__ == "__main__":
    # Test the video depth processor
    processor = VideoDepthProcessor()
    
    # Process it
    scene_pcd = processor.process_video_to_scene(
        SAMPLE_VIDEO, 
        output_path="reconstructed_scene.ply"
    )
    
    # Visualize
    o3d.visualization.draw_geometries([scene_pcd])