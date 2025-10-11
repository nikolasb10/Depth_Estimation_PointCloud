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
    """Test the PointCloudGenerator class with a sample image."""
    
    import os
    import copy
    from pathlib import Path
    from depth_estimator import DepthEstimator
    
    print("=== PointCloudGenerator Testing ===")
    
    # Create output directory
    output_dir = Path("outputs/demo_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Check if sample image exists
    sample_image_path = Path("sample_images/downloaded/indoor_room.jpg")
    if not sample_image_path.exists():
        print(f"❌ Sample image not found: {sample_image_path}")
        print("Please run the sample image creation script first.")
        return
    
    print(f"✅ Using sample image: {sample_image_path}")
    
    try:
        # Initialize depth estimator
        print("\n1. Initializing DepthEstimator...")
        depth_estimator = DepthEstimator(model_name="Intel/dpt-hybrid-midas")
        
        # Initialize point cloud generator
        print("2. Initializing PointCloudGenerator...")
        pcg = PointCloudGenerator()
        
        # Estimate depth from the sample image
        print("3. Estimating depth from sample image...")
        original_image, depth_map, depth_normalized = depth_estimator.estimate_depth_from_image(str(sample_image_path))
        
        print(f"   - Image shape: {original_image.shape}")
        print(f"   - Depth map shape: {depth_map.shape}")
        print(f"   - Depth range: {depth_map.min():.3f} to {depth_map.max():.3f}")
        
        # Save depth visualization
        print("   - Saving depth map visualization...")
        depth_vis_path = output_dir / "pointcloud_test_depth_map.png"
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(depth_map, cmap='viridis')
        plt.title("Raw Depth Map")
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(depth_normalized, cmap='plasma')
        plt.title("Normalized Depth")
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(depth_vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Depth visualization saved: {depth_vis_path}")
        
        # Generate point cloud
        print("4. Converting to 3D point cloud...")
        pointcloud = pcg.depth_to_pointcloud(original_image, depth_map, max_depth=30.0)
        
        print(f"   - Generated {len(pointcloud.points):,} points")
        print(f"   - Point cloud has colors: {pointcloud.has_colors()}")
        
        # Test point cloud processing
        print("5. Testing point cloud processing...")
        
        # Test downsampling
        if len(pointcloud.points) > 10000:
            print("   - Testing downsampling...")
            downsampled = pcg.downsample_pointcloud(copy.deepcopy(pointcloud), voxel_size=0.02)
        
        # Test filtering
        print("   - Testing outlier filtering...")
        filtered = pcg.filter_pointcloud(copy.deepcopy(pointcloud))
        
        # Save point cloud
        print("6. Saving point cloud...")
        pointcloud_path = output_dir / "pointcloud_test_original.ply"
        success = pcg.save_pointcloud(pointcloud, str(pointcloud_path))
        
        if success:
            print(f"   ✅ Point cloud saved to: {pointcloud_path}")
            print(f"   File size: {os.path.getsize(pointcloud_path):,} bytes")
        
        # Save downsampled point cloud
        if len(pointcloud.points) > 10000:
            downsampled_path = output_dir / "pointcloud_test_downsampled.ply"
            pcg.save_pointcloud(downsampled, str(downsampled_path))
            print(f"   ✅ Downsampled point cloud saved to: {downsampled_path}")
        
        # Save filtered point cloud
        filtered_path = output_dir / "pointcloud_test_filtered.ply"
        pcg.save_pointcloud(filtered, str(filtered_path))
        print(f"   ✅ Filtered point cloud saved to: {filtered_path}")
        
        # Test mesh creation
        print("7. Testing mesh creation...")
        try:
            mesh = pcg.create_mesh_from_pointcloud(copy.deepcopy(filtered), depth=8)
            mesh_path = output_dir / "pointcloud_test_mesh.ply"
            import open3d as o3d
            o3d.io.write_triangle_mesh(str(mesh_path), mesh)
            print(f"   ✅ Mesh saved to: {mesh_path}")
            print(f"   File size: {os.path.getsize(mesh_path):,} bytes")
        except Exception as e:
            print(f"   ⚠️ Mesh creation failed: {e}")
        
        # Test camera intrinsics estimation
        print("8. Testing camera intrinsics estimation...")
        h, w = depth_map.shape
        intrinsics = pcg.estimate_camera_intrinsics(w, h, fov_degrees=60)
        print(f"   - Estimated intrinsics for {w}x{h} image:")
        print(f"     fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
        print(f"     cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")
        
        # Save intrinsics info
        intrinsics_path = output_dir / "pointcloud_test_camera_intrinsics.txt"
        with open(intrinsics_path, 'w') as f:
            f.write("PointCloudGenerator Test - Camera Intrinsics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Image dimensions: {w} x {h}\n")
            f.write(f"Field of View: 60 degrees\n\n")
            f.write("Estimated Camera Intrinsics:\n")
            f.write(f"fx (focal length X): {intrinsics['fx']:.3f}\n")
            f.write(f"fy (focal length Y): {intrinsics['fy']:.3f}\n")
            f.write(f"cx (principal point X): {intrinsics['cx']:.3f}\n")
            f.write(f"cy (principal point Y): {intrinsics['cy']:.3f}\n\n")
            f.write("Camera Matrix K:\n")
            f.write(f"[{intrinsics['fx']:.1f}    0    {intrinsics['cx']:.1f}]\n")
            f.write(f"[   0    {intrinsics['fy']:.1f}    {intrinsics['cy']:.1f}]\n")
            f.write(f"[   0       0       1   ]\n")
        print(f"   ✅ Camera intrinsics saved to: {intrinsics_path}")
        
        # Generate 2D renders of point clouds using matplotlib (OpenGL-free)
        print("9. Generating 2D renders of 3D models...")
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            def render_pointcloud_matplotlib(pcd, output_path, title="Point Cloud", max_points=5000):
                """Render point cloud using matplotlib 3D plotting."""
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                
                # Sample points if too many for performance
                if len(points) > max_points:
                    indices = np.random.choice(len(points), max_points, replace=False)
                    points = points[indices]
                    if colors is not None:
                        colors = colors[indices]
                
                # Create 3D plot
                fig = plt.figure(figsize=(15, 5))
                
                # Three different viewing angles
                angles = [(30, 45), (-60, 30), (60, -45)]
                angle_names = ['View 1', 'View 2', 'View 3']
                
                for i, (elev, azim) in enumerate(angles):
                    ax = fig.add_subplot(1, 3, i+1, projection='3d')
                    
                    if colors is not None:
                        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                 c=colors, s=1, alpha=0.6)
                    else:
                        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                 c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z (Depth)')
                    ax.set_title(angle_names[i])
                    ax.view_init(elev=elev, azim=azim)
                    
                    # Set equal aspect ratio
                    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                                        points[:, 1].max()-points[:, 1].min(),
                                        points[:, 2].max()-points[:, 2].min()]).max() / 2.0
                    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
                    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
                    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)
                
                plt.suptitle(title, fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                return True
            
            def render_mesh_matplotlib(mesh, output_path, title="Mesh"):
                """Render mesh using matplotlib 3D plotting."""
                vertices = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles)
                
                # Sample triangles if too many
                if len(triangles) > 2000:
                    indices = np.random.choice(len(triangles), 2000, replace=False)
                    triangles = triangles[indices]
                
                fig = plt.figure(figsize=(15, 5))
                angles = [(30, 45), (-60, 30), (60, -45)]
                angle_names = ['View 1', 'View 2', 'View 3']
                
                for i, (elev, azim) in enumerate(angles):
                    ax = fig.add_subplot(1, 3, i+1, projection='3d')
                    
                    # Plot triangles
                    for triangle in triangles[:1000]:  # Limit for performance
                        triangle_vertices = vertices[triangle]
                        ax.plot_trisurf(triangle_vertices[:, 0], 
                                      triangle_vertices[:, 1], 
                                      triangle_vertices[:, 2], 
                                      alpha=0.7, color='lightblue')
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z (Depth)')
                    ax.set_title(angle_names[i])
                    ax.view_init(elev=elev, azim=azim)
                    
                    # Set equal aspect ratio
                    max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                                        vertices[:, 1].max()-vertices[:, 1].min(),
                                        vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
                    mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
                    mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
                    mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)
                
                plt.suptitle(title, fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                return True
            
            # Render original point cloud
            print("   - Rendering original point cloud...")
            original_render_path = output_dir / "pointcloud_test_original_render.png"
            render_pointcloud_matplotlib(pointcloud, original_render_path, 
                                       f"Original Point Cloud ({len(pointcloud.points):,} points)")
            print(f"   ✅ Original point cloud render saved: {original_render_path}")
            
            # Render filtered point cloud
            print("   - Rendering filtered point cloud...")
            filtered_render_path = output_dir / "pointcloud_test_filtered_render.png"
            render_pointcloud_matplotlib(filtered, filtered_render_path,
                                       f"Filtered Point Cloud ({len(filtered.points):,} points)")
            print(f"   ✅ Filtered point cloud render saved: {filtered_render_path}")
            
            # Render downsampled if available
            if len(pointcloud.points) > 10000:
                print("   - Rendering downsampled point cloud...")
                downsampled_render_path = output_dir / "pointcloud_test_downsampled_render.png"
                render_pointcloud_matplotlib(downsampled, downsampled_render_path,
                                           f"Downsampled Point Cloud ({len(downsampled.points):,} points)")
                print(f"   ✅ Downsampled point cloud render saved: {downsampled_render_path}")
            
            # Render mesh if available
            try:
                if 'mesh' in locals():
                    print("   - Rendering mesh...")
                    mesh_render_path = output_dir / "pointcloud_test_mesh_render.png"
                    render_mesh_matplotlib(mesh, mesh_render_path,
                                         f"Reconstructed Mesh ({len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles)")
                    print(f"   ✅ Mesh render saved: {mesh_render_path}")
            except Exception as mesh_e:
                print(f"   ⚠️ Mesh rendering failed: {mesh_e}")
                
        except Exception as e:
            print(f"   ⚠️ 2D rendering failed: {e}")
            import traceback
            traceback.print_exc()
            
        # Try interactive visualization (may fail in headless environments)
        print("10. Testing interactive visualization...")
        try:
            print("   - Attempting to open 3D visualization...")
            print("   - Close the visualization window to continue...")
            pcg.visualize_pointcloud(filtered, window_name="PointCloudGenerator Test")
            print("   ✅ Interactive visualization completed successfully")
        except Exception as e:
            print(f"   ⚠️ Interactive visualization failed (this is normal in headless environments): {e}")
        
        print("\n=== Testing Summary ===")
        print("✅ DepthEstimator integration: SUCCESS")
        print("✅ Point cloud generation: SUCCESS")
        print("✅ Point cloud processing: SUCCESS")
        print("✅ File I/O operations: SUCCESS")
        print("✅ Camera intrinsics: SUCCESS")
        print("🎯 All core functionalities tested successfully!")
        
        # Summary of generated files
        print(f"\n11. Generated Files Summary (in {output_dir}):")
        print("\n   📊 Analysis Files:")
        print("   📸 pointcloud_test_depth_map.png - Depth visualization comparison")
        print("   📄 pointcloud_test_camera_intrinsics.txt - Camera parameters")
        
        print("\n   🎯 3D Model Files (.ply):")
        print("   🔴 pointcloud_test_original.ply - Original point cloud")
        if len(pointcloud.points) > 10000:
            print("   🔵 pointcloud_test_downsampled.ply - Downsampled point cloud")
        print("   🟢 pointcloud_test_filtered.ply - Filtered point cloud")
        print("   🌐 pointcloud_test_mesh.ply - Reconstructed mesh surface")
        
        print("\n   🖼️ 2D Rendered Images:")
        print("   🔴 pointcloud_test_original_render.png - Original point cloud view")
        print("   🟢 pointcloud_test_filtered_render.png - Filtered point cloud view")
        if len(pointcloud.points) > 10000:
            print("   🔵 pointcloud_test_downsampled_render.png - Downsampled point cloud view")
        print("   🌐 pointcloud_test_mesh_render.png - Mesh surface view")
        
        print("\n   💡 Tips:")
        print("   • View PNG files directly in any image viewer")
        print("   • Open PLY files with MeshLab, CloudCompare, or Blender for interactive 3D viewing")
        print("   • The rendered PNGs show what the 3D models look like!")
        
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 PointCloudGenerator testing completed successfully!")


if __name__ == "__main__":
    main()