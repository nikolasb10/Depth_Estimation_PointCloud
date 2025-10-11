"""
Streamlit component for Point Cloud Visualization functionality.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import time
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

from pointcloud_generator import PointCloudGenerator
from depth_estimator import DepthEstimator


def create_3d_plot_plotly(pointcloud, title="3D Point Cloud", max_points=10000):
    """Create interactive 3D plot of point cloud using Plotly."""
    try:
        import open3d as o3d
        
        points = np.asarray(pointcloud.points)
        if len(points) == 0:
            return None
        
        # Sample points for performance if too many
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        
        # Get colors if available
        if pointcloud.has_colors():
            colors = np.asarray(pointcloud.colors)
            if len(colors) > max_points:
                colors = colors[indices]
            # Convert to RGB strings for plotly
            color_strings = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors]
        else:
            # Use depth-based coloring
            depths = points[:, 2]  # Z-coordinate as depth
            color_strings = None
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1], 
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=color_strings if color_strings else depths,
                colorscale='Viridis' if not color_strings else None,
                opacity=0.8
            ),
            name='Point Cloud'
        )])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (meters)",
                yaxis_title="Y (meters)", 
                zaxis_title="Z (depth meters)",
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Failed to create 3D plot: {str(e)}")
        return None


def render_point_cloud_component():
    """Main function to render the Point Cloud Visualization component."""
    st.header("🎯 Point Cloud Visualization")
    st.markdown("""
    Generate interactive 3D point clouds from RGB images and depth maps using advanced computer vision techniques.
    """)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; text-align: center;">
            <h4>🔍 DEPTH ESTIMATION</h4>
            <p>State-of-the-art DPT/MiDaS models<br/>
            Convert 2D images to depth maps</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; text-align: center;">
            <h4>🎯 3D RECONSTRUCTION</h4>
            <p>Camera intrinsics projection<br/>
            Convert depth maps to 3D coordinates</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; text-align: center;">
            <h4>⚡ PROCESSING</h4>
            <p>Filtering & Downsampling<br/>
            Mesh generation & Export</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📸 Single Image Processing", "🌐 Point Cloud Analysis", "⚙️ Technical Details"])
    
    with tab1:
        render_single_image_tab()
    
    with tab2:
        render_analysis_tab()
    
    with tab3:
        render_technical_details_tab()


def render_single_image_tab():
    """Render the single image processing tab."""
    st.header("Single Image Point Cloud Generation")
    
    # Get selected model from session state (set in main app)
    selected_model = st.session_state.get('selected_model', 'Intel/dpt-hybrid-midas')
    
    # Configuration in columns
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("⚙️ Processing Settings")
        
        # Point cloud settings
        max_depth = st.slider("Max Depth (meters)", 1.0, 50.0, 15.0, 0.5)
        voxel_size = st.slider("Downsampling Voxel Size", 0.01, 0.5, 0.05, 0.01)
        
        # Camera settings
        fov_degrees = st.slider("Field of View (degrees)", 30, 120, 60, 5)
        
        # Processing options
        enable_downsampling = st.checkbox("Enable Downsampling", True)
        enable_filtering = st.checkbox("Enable Outlier Filtering", True)
        enable_mesh_generation = st.checkbox("Generate Mesh Surface", False)
        
        if enable_filtering:
            filter_neighbors = st.slider("Filter Neighbors", 10, 50, 20, 5)
            filter_std_ratio = st.slider("Filter Std Ratio", 1.0, 5.0, 2.0, 0.1)
        
        if enable_mesh_generation:
            mesh_depth = st.slider("Mesh Reconstruction Depth", 6, 12, 9, 1)
    
    with col1:
        # Image input section
        st.subheader("🖼️ Image Input")
        
        uploaded_file = st.file_uploader(
            "Upload an image for point cloud generation",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a PNG, JPG, or JPEG image"
        )
        
        # Sample images section
        st.markdown("**Or choose from sample images:**")
        col1_1, col2_1, col3_1, col4_1 = st.columns(4)
        
        sample_images = {
            "Indoor Room": "sample_images/downloaded/indoor_room.jpg",
            "Outdoor Landscape": "sample_images/downloaded/outdoor_landscape.jpg", 
            "Street Scene": "sample_images/downloaded/street_scene.jpg",
            "Geometric Shapes": "sample_images/synthetic/geometric_shapes.png"
        }
        
        # Use session state to persist selected sample
        if 'selected_sample' not in st.session_state:
            st.session_state.selected_sample = None
            
        with col1_1:
            if st.button("🏠 Indoor"):
                st.session_state.selected_sample = "Indoor Room"
                st.rerun()
        with col2_1:
            if st.button("🌄 Outdoor"):
                st.session_state.selected_sample = "Outdoor Landscape"
                st.rerun()
        with col3_1:
            if st.button("🚗 Street"):
                st.session_state.selected_sample = "Street Scene"
                st.rerun()
        with col4_1:
            if st.button("🔺 Geometric"):
                st.session_state.selected_sample = "Geometric Shapes"
                st.rerun()
                
        selected_sample = st.session_state.selected_sample
        
        # Show current selection and clear button
        if selected_sample:
            st.info(f"Selected: {selected_sample}")
            if st.button("🗑️ Clear Selection"):
                st.session_state.selected_sample = None
                st.rerun()
    
    # Process image    
    if uploaded_file is not None or selected_sample is not None:
        # Load image
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.success("✅ Image uploaded successfully!")
        else:
            sample_path = Path(sample_images[selected_sample])
            if sample_path.exists():
                image = Image.open(sample_path).convert("RGB")
            else:
                st.error("L Sample images not found. Please run the setup first:")
                st.code("uv run examples/create_sample_images.py")
                return
        
        # Display original image
        st.subheader("🧩 Original Image")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)
        with col2:
            st.metric("Resolution", f"{image.size[0]}x{image.size[1]}")
            st.metric("Aspect Ratio", f"{image.size[0]/image.size[1]:.2f}")
            st.metric("Channels", "RGB (3)")
        
        # Process depth estimation and point cloud generation        
        if st.button("🌀 Generate Point Cloud", type="primary"):
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            # Process with visible progress
            try:
                # Initialize models
                status_placeholder.text("🔧 Initializing models...")
                depth_estimator = DepthEstimator(model_name=selected_model)
                pcg = PointCloudGenerator()
                progress_bar.progress(10)
                
                # Convert PIL to numpy
                image_np = np.array(image)
                    
                # Save temporarily for depth estimation
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    
                    # Estimate depth
                    st.text("🔍 Estimating depth...")
                    original_image, depth_map, depth_normalized = depth_estimator.estimate_depth_from_image(tmp_file.name)
                    progress_bar.progress(30)
                    
                    # Set camera intrinsics
                    h, w = depth_map.shape[:2]
                    pcg.estimate_camera_intrinsics(w, h, fov_degrees)
                    
                    # Generate point cloud
                    st.text("🌀 Generating point cloud...")
                    pointcloud = pcg.depth_to_pointcloud(image_np, depth_map, max_depth)
                    original_point_count = len(pointcloud.points)
                    progress_bar.progress(80)

                    # Store results in session state
                    processing_results = {
                        'original_image': original_image,
                        'depth_map': depth_map,
                        'depth_normalized': depth_normalized,
                        'pointcloud': pointcloud,
                        'original_point_count': original_point_count,
                        'camera_intrinsics': pcg.camera_intrinsics
                    }
                    
                    # Apply processing steps
                    current_pointcloud = pointcloud
                    
                    # Downsampling
                    if enable_downsampling and len(current_pointcloud.points) > 10000:
                        st.text("📉 Downsampling point cloud...")
                        progress_bar.progress(90)
                        current_pointcloud = pcg.downsample_pointcloud(current_pointcloud, voxel_size)
                        processing_results['downsampled_pointcloud'] = current_pointcloud
                        processing_results['downsampled_point_count'] = len(current_pointcloud.points)
                    
                    # Filtering
                    if enable_filtering:
                        st.text("🧹 Filtering outliers...")
                        progress_bar.progress(100)
                        current_pointcloud = pcg.filter_pointcloud(
                            current_pointcloud, 
                            nb_neighbors=filter_neighbors if enable_filtering else 20, 
                            std_ratio=filter_std_ratio if enable_filtering else 2.0
                        )
                        processing_results['filtered_pointcloud'] = current_pointcloud
                        processing_results['filtered_point_count'] = len(current_pointcloud.points)
                    
                    # Mesh generation
                    if enable_mesh_generation:
                        st.text("🧱 Generating mesh...")
                        try:
                            mesh = pcg.create_mesh_from_pointcloud(current_pointcloud.copy(), depth=mesh_depth if enable_mesh_generation else 9)
                            processing_results['mesh'] = mesh
                            processing_results['mesh_vertex_count'] = len(mesh.vertices)
                            processing_results['mesh_triangle_count'] = len(mesh.triangles)
                        except Exception as mesh_e:
                            st.warning(f"� Mesh generation failed: {mesh_e}")
                    
                    processing_results['final_pointcloud'] = current_pointcloud
                    st.session_state.pointcloud_results = processing_results
                    
                    # Clean up
                    os.unlink(tmp_file.name)
                    
            except Exception as e:
                st.error(f"L Processing failed: {str(e)}")
                return
            
            st.success("🎉 Point cloud generation completed!")
    
    # Display results if available
    if st.session_state.get('pointcloud_results') is not None:
        results = st.session_state.pointcloud_results
        
        st.subheader("📊 Processing Results")
        
        # Show depth maps
        col1, col2 = st.columns(2)
        with col1:
            st.image(results['depth_normalized'], caption="Depth Map (Normalized)", use_container_width=True)
        with col2:
            # Create colored depth map
            depth_colored = cv2.applyColorMap(results['depth_normalized'], cv2.COLORMAP_PLASMA)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            st.image(depth_colored, caption="Depth Map (Colored)", use_container_width=True)
        
        # Processing summary
        st.subheader("📈 Processing Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original Points", f"{results['original_point_count']:,}")
        with col2:
            if 'downsampled_point_count' in results:
                st.metric("After Downsampling", f"{results['downsampled_point_count']:,}")
        with col3:
            if 'filtered_point_count' in results:
                st.metric("After Filtering", f"{results['filtered_point_count']:,}")
        with col4:
            final_count = len(results['final_pointcloud'].points)
            st.metric("Final Points", f"{final_count:,}")
        
        # 3D visualization
        st.subheader("🌐 3D Point Cloud Visualization")
        
        viz_option = st.selectbox(
            "Select point cloud to visualize:",
            ["Final Processed", "Original", "Downsampled", "Filtered"] if 'downsampled_pointcloud' in results else ["Final Processed", "Original"]
        )
        
        # Select pointcloud based on option
        if viz_option == "Original":
            selected_pointcloud = results['pointcloud']
            title = f"Original Point Cloud ({results['original_point_count']:,} points)"
        elif viz_option == "Downsampled" and 'downsampled_pointcloud' in results:
            selected_pointcloud = results['downsampled_pointcloud']
            title = f"Downsampled Point Cloud ({results['downsampled_point_count']:,} points)"
        elif viz_option == "Filtered" and 'filtered_pointcloud' in results:
            selected_pointcloud = results['filtered_pointcloud']
            title = f"Filtered Point Cloud ({results['filtered_point_count']:,} points)"
        else:
            selected_pointcloud = results['final_pointcloud']
            title = f"Final Point Cloud ({len(results['final_pointcloud'].points):,} points)"
        
        # Create and display 3D plot
        fig = create_3d_plot_plotly(selected_pointcloud, title)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Download options
        st.subheader("⬇️ Download Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download point cloud
            if st.button("⬇️ Download Point Cloud (.ply)"):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_ply:
                    import open3d as o3d
                    o3d.io.write_point_cloud(tmp_ply.name, selected_pointcloud)
                    with open(tmp_ply.name, 'rb') as f:
                        ply_data = f.read()
                    os.unlink(tmp_ply.name)
                    
                    st.download_button(
                        label="Download PLY file",
                        data=ply_data,
                        file_name=f"pointcloud_{viz_option.lower().replace(' ', '_')}.ply",
                        mime="application/octet-stream"
                    )
        
        with col2:
            # Download depth map
            depth_buffer = cv2.imencode('.png', results['depth_normalized'])[1].tobytes()
            st.download_button(
                label="⬇️ Download Depth Map",
                data=depth_buffer,
                file_name="depth_map.png",
                mime="image/png"
            )
        
        with col3:
            # Download mesh if available
            if 'mesh' in results:
                if st.button("⬇️ Download Mesh (.ply)"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_mesh:
                        import open3d as o3d
                        o3d.io.write_triangle_mesh(tmp_mesh.name, results['mesh'])
                        with open(tmp_mesh.name, 'rb') as f:
                            mesh_data = f.read()
                        os.unlink(tmp_mesh.name)
                        
                        st.download_button(
                            label="Download Mesh PLY",
                            data=mesh_data,
                            file_name="reconstructed_mesh.ply",
                            mime="application/octet-stream"
                        )


def render_analysis_tab():
    """Render the point cloud analysis tab."""
    st.header("🌐 Point Cloud Analysis")
    
    if st.session_state.get('pointcloud_results') is None:
        st.info("ℹ️ Process an image first to see analysis here.")
        return
    
    results = st.session_state.pointcloud_results
    pointcloud = results['final_pointcloud']
    
    # Basic statistics
    st.subheader("📊 Point Cloud Statistics")
    
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors) if pointcloud.has_colors() else None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Points", f"{len(points):,}")
        st.metric("Has Colors", "Yes" if colors is not None else "No")
        
        # Bounding box
        bounds = pointcloud.get_axis_aligned_bounding_box()
        extent = bounds.get_extent()
        st.metric("Bounding Box (X×Y×Z)", f"{extent[0]:.2f}×{extent[1]:.2f}×{extent[2]:.2f}m")
    
    with col2:
        # Coordinate statistics
        st.markdown("**Coordinate Ranges:**")
        st.text(f"X: {points[:, 0].min():.2f} to {points[:, 0].max():.2f}")
        st.text(f"Y: {points[:, 1].min():.2f} to {points[:, 1].max():.2f}")
        st.text(f"Z: {points[:, 2].min():.2f} to {points[:, 2].max():.2f}")
        
        # Center of mass
        center = points.mean(axis=0)
        st.text(f"Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    
    with col3:
        # Processing chain summary
        st.markdown("**Processing Chain:**")
        st.text(f"Original: {results['original_point_count']:,} points")
        if 'downsampled_point_count' in results:
            st.text(f"Downsampled: {results['downsampled_point_count']:,} points")
        if 'filtered_point_count' in results:
            st.text(f"Filtered: {results['filtered_point_count']:,} points")
        st.text(f"Final: {len(points):,} points")
    
    # Depth distribution
    st.subheader("📉 Depth Distribution")
    
    depths = points[:, 2]
    fig_hist = go.Figure(data=[go.Histogram(x=depths, nbinsx=50)])
    fig_hist.update_layout(
        title="Distribution of Depth Values (Z-coordinate)",
        xaxis_title="Depth (meters)",
        yaxis_title="Number of Points",
        height=400
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Camera intrinsics info
    if 'camera_intrinsics' in results:
        st.subheader("🎯 Camera Intrinsics")
        intrinsics = results['camera_intrinsics']
        
        col1, col2 = st.columns(2)
        with col1:
            st.json({
                "Focal Length X (fx)": f"{intrinsics['fx']:.2f}",
                "Focal Length Y (fy)": f"{intrinsics['fy']:.2f}",
                "Principal Point X (cx)": f"{intrinsics['cx']:.2f}",
                "Principal Point Y (cy)": f"{intrinsics['cy']:.2f}"
            })
        
        with col2:
            st.json({
                "Image Width": intrinsics['width'],
                "Image Height": intrinsics['height'],
                "Aspect Ratio": f"{intrinsics['width']/intrinsics['height']:.3f}"
            })


def render_technical_details_tab():
    """Render the technical details tab."""
    st.header("⚙️ Technical Implementation Details")    

    st.markdown("""
    ### =
 Depth Estimation to Point Cloud Pipeline
    
    **1. Monocular Depth Estimation:**
    - Uses state-of-the-art DPT (Dense Prediction Transformer) models
    - Converts single RGB images to dense depth maps
    - Models trained on diverse datasets (NYU Depth V2, KITTI, etc.)
    """)
    
    st.code("""
# Depth estimation using transformers
depth_estimator = DepthEstimator(model_name="Intel/dpt-hybrid-midas")
original_image, depth_map, depth_normalized = depth_estimator.estimate_depth_from_image(image_path)
    """, language="python")
    
    st.markdown("""
    **2. Camera Intrinsics Estimation:**
    - Estimates focal length from field of view
    - Assumes principal point at image center
    - Supports custom intrinsics for calibrated cameras
    """)
    
    st.code("""
# Camera intrinsics estimation
fov_rad = np.radians(fov_degrees)
fx = image_width / (2 * np.tan(fov_rad / 2))
fy = fx  # Assume square pixels
cx = image_width / 2  # Principal point X
cy = image_height / 2  # Principal point Y
    """, language="python")
    
    st.markdown("""
    **3. 3D Point Cloud Generation:**
    - Converts 2D pixels to 3D coordinates using camera geometry
    - Applies depth scaling and filtering
    - Preserves RGB color information from original image
    """)
    
    st.code("""
# Convert 2D + depth to 3D coordinates
z = depth_normalized[valid_mask]
x = (i[valid_mask] - cx) * z / fx
y = (j[valid_mask] - cy) * z / fy

# Stack into 3D points with colors
points_3d = np.stack([x, y, z], axis=-1)
colors = rgb_image[valid_mask] / 255.0
    """, language="python")
    
    st.markdown("""
    ### � Point Cloud Processing Pipeline
    
    **Downsampling (Voxel Grid):**
    - Reduces point density for performance
    - Preserves overall shape and structure
    - Configurable voxel size parameter
    
    **Outlier Filtering (Statistical):**
    - Removes isolated points that are likely noise
    - Uses local neighborhood statistics
    - Configurable neighbor count and standard deviation threshold
    
    **Mesh Generation (Poisson Reconstruction):**
    - Creates continuous surface from point cloud
    - Estimates surface normals automatically
    - Configurable reconstruction depth for detail level
    """)
    
    st.markdown("""
    ### <� Key Algorithms & Techniques
    
    | Component | Algorithm | Purpose |
    |-----------|-----------|---------|
    | **Depth Estimation** | DPT/MiDaS Transformer | Monocular depth from RGB |
    | **3D Projection** | Pinhole camera model | 2D�3D coordinate transformation |
    | **Downsampling** | Voxel grid filtering | Reduce point density |
    | **Outlier Removal** | Statistical filtering | Remove noise points |
    | **Surface Reconstruction** | Poisson reconstruction | Point cloud � mesh |
    | **Visualization** | Plotly 3D scatter | Interactive 3D viewing |
    
    ### =� Applications
    
    This pipeline enables various computer vision applications:
    
     **Robotics**: Navigation, obstacle avoidance, SLAM  
     **Autonomous Vehicles**: 3D scene understanding  
     **AR/VR**: Depth-aware rendering and occlusion  
     **3D Modeling**: Single-view reconstruction  
     **Medical Imaging**: Volumetric analysis  
     **Architecture**: Space planning and visualization  
    
    ### =, Technical Specifications
    
    **Input Requirements:**
    - RGB images (PNG, JPG, JPEG)
    - Resolution: Any (automatically handled)
    - Color space: RGB or grayscale
    
    **Output Formats:**
    - Point clouds: PLY format (Open3D compatible)
    - Meshes: PLY format with triangulated surfaces  
    - Depth maps: PNG images (normalized 0-255)
    - Visualization: Interactive Plotly 3D plots
    
    **Performance Considerations:**
    - Memory usage scales with image resolution
    - Processing time depends on model complexity
    - Large point clouds benefit from downsampling
    - Interactive visualization limited to ~10K points
    """)