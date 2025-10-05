"""
Point cloud generation and visualization from depth maps and RGB images.
"""

import numpy as np
import open3d as o3d
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class PointCloudGenerator:
    """
    Generate and visualize 3D point clouds from RGB images and depth maps.
    """
    
    def __init__(self, camera_intrinsics=None):
        """
        Initialize point cloud generator.
        
        Args:
            camera_intrinsics (dict): Camera intrinsic parameters
                - fx, fy: focal lengths
                - cx, cy: principal point coordinates
                - width, height: image dimensions
        """
        self.camera_intrinsics = camera_intrinsics
        
    def set_camera_intrinsics(self, fx, fy, cx, cy, width, height):
        """
        Set camera intrinsic parameters.
        
        Args:
            fx, fy (float): Focal lengths in pixels
            cx, cy (float): Principal point coordinates
            width, height (int): Image dimensions
        """
        self.camera_intrinsics = {
            'fx': fx, 'fy': fy,
            'cx': cx, 'cy': cy,
            'width': width, 'height': height
        }
    
    def estimate_camera_intrinsics(self, image_width, image_height, fov_degrees=60):
        """
        Estimate camera intrinsics assuming a typical camera setup.
        
        Args:
            image_width (int): Image width in pixels
            image_height (int): Image height in pixels
            fov_degrees (float): Horizontal field of view in degrees
            
        Returns:
            dict: Estimated camera intrinsics
        """
        # Convert FOV to radians
        fov_rad = np.radians(fov_degrees)
        
        # Estimate focal length
        fx = image_width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Assume square pixels
        
        # Principal point at image center
        cx = image_width / 2
        cy = image_height / 2
        
        intrinsics = {
            'fx': fx, 'fy': fy,
            'cx': cx, 'cy': cy,
            'width': image_width, 'height': image_height
        }
        
        self.camera_intrinsics = intrinsics
        return intrinsics
    
    def depth_to_pointcloud(self, rgb_image, depth_map, max_depth=10.0):
        """
        Convert RGB image and depth map to 3D point cloud.
        
        Args:
            rgb_image (np.ndarray): RGB image array
            depth_map (np.ndarray): Depth map array
            max_depth (float): Maximum depth value to include in point cloud
            
        Returns:
            o3d.geometry.PointCloud: Open3D point cloud object
        """
        if self.camera_intrinsics is None:
            # Estimate intrinsics if not provided
            h, w = depth_map.shape[:2]
            self.estimate_camera_intrinsics(w, h)
        
        # Get camera parameters
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        
        # Create coordinate grids
        height, width = depth_map.shape
        i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
        
        # Convert depth values (assuming they are in relative units, normalize to meters)
        # MiDaS outputs relative depth, so we normalize and scale
        depth_normalized = depth_map / depth_map.max() * max_depth
        
        # Filter out invalid depths
        valid_mask = (depth_normalized > 0) & (depth_normalized < max_depth)
        
        # Convert to 3D coordinates
        z = depth_normalized[valid_mask]
        x = (i[valid_mask] - cx) * z / fx
        y = (j[valid_mask] - cy) * z / fy
        
        # Stack coordinates
        points_3d = np.stack([x, y, z], axis=-1)
        
        # Get corresponding colors
        if len(rgb_image.shape) == 3:
            colors = rgb_image[valid_mask] / 255.0  # Normalize to [0, 1]
        else:
            # If grayscale, create RGB
            gray_colors = rgb_image[valid_mask] / 255.0
            colors = np.stack([gray_colors, gray_colors, gray_colors], axis=-1)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def visualize_pointcloud(self, pointcloud, window_name="Point Cloud", 
                           show_coordinate_frame=True, point_size=1.0):
        """
        Visualize point cloud using Open3D.
        
        Args:
            pointcloud (o3d.geometry.PointCloud): Point cloud to visualize
            window_name (str): Window title
            show_coordinate_frame (bool): Whether to show coordinate frame
            point_size (float): Size of points in visualization
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        
        # Add point cloud
        vis.add_geometry(pointcloud)
        
        # Add coordinate frame if requested
        if show_coordinate_frame:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            vis.add_geometry(coordinate_frame)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        # Run visualization
        vis.run()
        vis.destroy_window()
    
    def save_pointcloud(self, pointcloud, filepath):
        """
        Save point cloud to file.
        
        Args:
            pointcloud (o3d.geometry.PointCloud): Point cloud to save
            filepath (str): Output file path (.ply, .pcd, .xyz)
        """
        success = o3d.io.write_point_cloud(filepath, pointcloud)
        if success:
            print(f"Point cloud saved to: {filepath}")
        else:
            print(f"Failed to save point cloud to: {filepath}")
        return success
    
    def downsample_pointcloud(self, pointcloud, voxel_size=0.05):
        """
        Downsample point cloud to reduce number of points.
        
        Args:
            pointcloud (o3d.geometry.PointCloud): Input point cloud
            voxel_size (float): Voxel size for downsampling
            
        Returns:
            o3d.geometry.PointCloud: Downsampled point cloud
        """
        downsampled = pointcloud.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled from {len(pointcloud.points)} to {len(downsampled.points)} points")
        return downsampled
    
    def filter_pointcloud(self, pointcloud, nb_neighbors=20, std_ratio=2.0):
        """
        Remove outlier points from point cloud.
        
        Args:
            pointcloud (o3d.geometry.PointCloud): Input point cloud
            nb_neighbors (int): Number of neighbors for outlier detection
            std_ratio (float): Standard deviation ratio threshold
            
        Returns:
            o3d.geometry.PointCloud: Filtered point cloud
        """
        filtered, _ = pointcloud.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        print(f"Filtered from {len(pointcloud.points)} to {len(filtered.points)} points")
        return filtered
    
    def create_mesh_from_pointcloud(self, pointcloud, depth=9):
        """
        Create a mesh surface from point cloud using Poisson reconstruction.
        
        Args:
            pointcloud (o3d.geometry.PointCloud): Input point cloud
            depth (int): Reconstruction depth (higher = more detail)
            
        Returns:
            o3d.geometry.TriangleMesh: Reconstructed mesh
        """
        # Estimate normals
        pointcloud.estimate_normals()
        
        # Poisson surface reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pointcloud, depth=depth
        )
        
        print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        return mesh


def main():
    """Example usage of the PointCloudGenerator class."""
    
    # Initialize point cloud generator
    pcg = PointCloudGenerator()
    
    print("PointCloudGenerator initialized successfully!")
    print("Use with depth_estimator.py to create point clouds from images.")
    
    # Example usage:
    # 1. Get depth map from DepthEstimator
    # 2. Use depth_to_pointcloud() to generate point cloud
    # 3. Visualize with visualize_pointcloud()
    # 4. Save with save_pointcloud()


if __name__ == "__main__":
    main()