"""
Streamlit web application for real-time depth estimation and point cloud visualization.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from depth_estimator import DepthEstimator
from pointcloud_generator import PointCloudGenerator
from model_benchmarker import ModelBenchmarker


# Configure Streamlit page
st.set_page_config(
    page_title="Depth Estimation & Point Cloud Viewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_depth_estimator(model_name):
    """Load and cache the depth estimation model."""
    return DepthEstimator(model_name=model_name)


def check_model_availability():
    """Check which models are available locally."""
    models_info = {
        "Intel/dpt-large": {"available": False, "size": "1.37GB", "description": "Highest quality, slower"},
        "Intel/dpt-hybrid-midas": {"available": False, "size": "490MB", "description": "Best balance"},
        "Intel/dpt-beit-base-384": {"available": False, "size": "350MB", "description": "Good alternative"}
    }
    
    # Check local cache
    project_root = Path(__file__).parent
    models_cache = project_root / "models_cache"
    
    if models_cache.exists():
        print(f"Models cache exists")
        for model_name in models_info.keys():
            cache_name = model_name.replace("/", "--")
            model_path = models_cache / f"models--{cache_name}"
            print(f"Checking for model at {model_path}")
            if model_path.exists():
                models_info[model_name]["available"] = True
    
    return models_info


@st.cache_resource
def load_pointcloud_generator():
    """Load and cache the point cloud generator."""
    return PointCloudGenerator()


def main():
    st.title("🔍 Depth Estimation & Point Cloud Visualization")
    st.markdown("""
    This application demonstrates real-time depth estimation from monocular images 
    and generates interactive 3D point clouds. Upload an image or use sample images 
    to see depth maps and 3D reconstructions.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Check model availability
    models_info = check_model_availability()
    
    # Model selection with availability status
    st.sidebar.subheader("🤖 Model Selection")
    
    # Show model status
    for model_name, info in models_info.items():
        status = "✅ Ready (Local)" if info["available"] else "⬇️ Will Download"
        st.sidebar.text(f"{status} - {info['description']} ({info['size']})")
    
    # Model selection dropdown
    model_options = {
        "Intel/dpt-hybrid-midas": "DPT-Hybrid (Best balance)",
        "Intel/dpt-large": "DPT-Large (Highest quality)",
        "Intel/dpt-beit-base-384": "DPT-BEiT (Good alternative)"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Depth Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )
    
    # Show download info if model not available locally
    if not models_info[selected_model]["available"]:
        st.sidebar.warning(f"⚠️ Model will be downloaded (~{models_info[selected_model]['size']}) on first use")
        
        if st.sidebar.button("📥 Download All Models Now"):
            with st.spinner("Downloading models... This may take several minutes."):
                # Run download script
                import subprocess
                result = subprocess.run([
                    sys.executable, "scripts/download_models.py"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.sidebar.success("✅ Models downloaded successfully!")
                    st.experimental_rerun()
                else:
                    st.sidebar.error(f"❌ Download failed: {result.stderr}")
    else:
        st.sidebar.success(f"✅ {selected_model} ready locally")
    
    # Point cloud settings
    st.sidebar.subheader("Point Cloud Settings")
    max_depth = st.sidebar.slider("Max Depth (meters)", 1.0, 20.0, 10.0, 0.5)
    voxel_size = st.sidebar.slider("Downsampling Voxel Size", 0.01, 0.2, 0.05, 0.01)
    point_size = st.sidebar.slider("Point Size", 0.5, 5.0, 1.0, 0.1)
    
    # Camera settings
    st.sidebar.subheader("Camera Settings")
    fov_degrees = st.sidebar.slider("Field of View (degrees)", 30, 120, 60, 5)
    
    # Load models
    with st.spinner("Loading models..."):
        depth_estimator = load_depth_estimator(selected_model)
        pcg = load_pointcloud_generator()
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Single Image", "📊 Model Comparison", "🎥 Webcam (Coming Soon)", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Image for Depth Estimation")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a PNG, JPG, or JPEG image"
        )
        
        # Sample images
        st.subheader("Or try sample images:")
        col1, col2, col3 = st.columns(3)
        
        sample_images = {
            "Indoor Scene": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800",
            "Outdoor Scene": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800", 
            "Street View": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800"
        }
        
        selected_sample = None
        with col1:
            if st.button("🏠 Indoor Scene"):
                selected_sample = "Indoor Scene"
        with col2:
            if st.button("🌲 Outdoor Scene"):
                selected_sample = "Outdoor Scene"
        with col3:
            if st.button("🛣️ Street View"):
                selected_sample = "Street View"
        
        # Process image
        if uploaded_file is not None or selected_sample is not None:
            
            # Load image
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.success("Image uploaded successfully!")
            else:
                # Load the selected sample image
                st.info(f"Loading sample: {selected_sample}")
                
                # Map sample names to actual image files
                sample_mapping = {
                    "Indoor Scene": "sample_images/downloaded/indoor_room.jpg",
                    "Outdoor Scene": "sample_images/downloaded/outdoor_landscape.jpg", 
                    "Street View": "sample_images/downloaded/street_scene.jpg"
                }
                
                # Alternative mapping if downloaded images don't exist
                synthetic_mapping = {
                    "Indoor Scene": "sample_images/synthetic/geometric_shapes.png",
                    "Outdoor Scene": "sample_images/synthetic/circular_pattern.png",
                    "Street View": "sample_images/synthetic/gradient_pattern.png"
                }
                
                # Try to load the selected sample image
                sample_path = Path(sample_mapping.get(selected_sample, ""))
                
                if not sample_path.exists():
                    # Fallback to synthetic images
                    sample_path = Path(synthetic_mapping.get(selected_sample, ""))
                
                if sample_path.exists():
                    image = Image.open(sample_path).convert("RGB")
                    st.success(f"Loaded sample: {selected_sample}")
                else:
                    st.error("Sample images not found. Please run the setup first:")
                    st.code("uv run examples/create_sample_images.py")
                    return
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, caption="Input Image", use_container_width=True)
            
            # Process depth estimation
            with st.spinner("Estimating depth..."):
                # Convert PIL to numpy
                image_np = np.array(image)
                
                # Save temporarily for depth estimation
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file.name)
                    
                    try:
                        # Estimate depth
                        original_image, depth_map, depth_normalized = depth_estimator.estimate_depth_from_image(tmp_file.name)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Depth Map (Normalized)")
                            st.image(depth_normalized, caption="Depth Map", use_container_width=True)
                        
                        with col2:
                            st.subheader("Depth Map (Raw)")
                            # Create a colored depth map
                            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
                            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
                            st.image(depth_colored, caption="Colored Depth Map", use_container_width=True)
                        
                        # Generate point cloud
                        st.subheader("3D Point Cloud Generation")
                        
                        with st.spinner("Generating point cloud..."):
                            # Set camera intrinsics
                            h, w = depth_map.shape[:2]
                            pcg.estimate_camera_intrinsics(w, h, fov_degrees)
                            
                            # Generate point cloud
                            pointcloud = pcg.depth_to_pointcloud(image_np, depth_map, max_depth)
                            
                            # Downsample for better performance
                            if len(pointcloud.points) > 50000:
                                pointcloud = pcg.downsample_pointcloud(pointcloud, voxel_size)
                            
                            # Filter outliers
                            pointcloud = pcg.filter_pointcloud(pointcloud)
                            
                            st.success(f"Generated point cloud with {len(pointcloud.points)} points")
                            
                            # Show point cloud statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Points", len(pointcloud.points))
                            with col2:
                                st.metric("Colors", len(pointcloud.colors))
                            with col3:
                                bounds = pointcloud.get_axis_aligned_bounding_box()
                                size = bounds.get_extent()
                                st.metric("Size (m)", f"{size[0]:.1f}×{size[1]:.1f}×{size[2]:.1f}")
                        
                        # Download options
                        st.subheader("Downloads")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Save depth map
                            depth_buffer = cv2.imencode('.png', depth_normalized)[1].tobytes()
                            st.download_button(
                                label="📥 Download Depth Map",
                                data=depth_buffer,
                                file_name="depth_map.png",
                                mime="image/png"
                            )
                        
                        with col2:
                            # Save point cloud
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_ply:
                                pcg.save_pointcloud(pointcloud, tmp_ply.name)
                                with open(tmp_ply.name, 'rb') as f:
                                    ply_data = f.read()
                                os.unlink(tmp_ply.name)
                                
                                st.download_button(
                                    label="📥 Download Point Cloud (.ply)",
                                    data=ply_data,
                                    file_name="pointcloud.ply",
                                    mime="application/octet-stream"
                                )
                        
                        with col3:
                            st.info("💡 Use MeshLab or CloudCompare to view .ply files")
                    
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_file.name)
    
    with tab2:
        st.header("📊 Model Performance Comparison")
        
        st.markdown("""
        Compare the performance of different depth estimation models on speed, memory usage, and quality.
        This helps you choose the best model for your specific use case.
        """)
        
        # Benchmark configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Benchmark Settings")
            
            # Model selection for comparison
            available_models = list(check_model_availability().keys())
            selected_models = st.multiselect(
                "Select models to compare",
                options=available_models,
                default=available_models[:2] if len(available_models) >= 2 else available_models,
                help="Choose which models to benchmark against each other"
            )
            
            # Test image selection
            num_test_images = st.slider(
                "Number of test images",
                min_value=1, max_value=10, value=3,
                help="More images = more accurate but slower benchmark"
            )
            
        with col2:
            st.subheader("📋 What gets measured:")
            st.markdown("""
            - **Inference Speed**: Time to process each image
            - **Memory Usage**: RAM consumption during processing  
            - **Model Loading**: Time to initialize the model
            - **FPS**: Frames per second capability
            - **Consistency**: Performance variance across images
            """)
        
        # Run benchmark button
        if st.button("🚀 Run Benchmark", type="primary", disabled=len(selected_models) < 1):
            if len(selected_models) < 1:
                st.error("Please select at least one model to benchmark")
            else:
                # Find test images
                sample_dirs = [Path("sample_images/synthetic"), Path("sample_images/downloaded")]
                test_images = []
                
                for sample_dir in sample_dirs:
                    if sample_dir.exists():
                        test_images.extend(list(sample_dir.glob("*.jpg")))
                        test_images.extend(list(sample_dir.glob("*.png")))
                
                if not test_images:
                    st.error("No test images found. Please run the sample image creation first.")
                    st.code("uv run examples/create_sample_images.py")
                else:
                    # Limit test images
                    test_images = [str(img) for img in test_images[:num_test_images]]
                    
                    st.info(f"Running benchmark on {len(test_images)} images with {len(selected_models)} models...")
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Initialize benchmarker
                        benchmarker = ModelBenchmarker()
                        
                        # Run benchmark
                        with st.spinner("Running benchmark... This may take several minutes"):
                            df = benchmarker.compare_models(test_images, selected_models)
                            progress_bar.progress(100)
                        
                        if not df.empty:
                            st.success("✅ Benchmark completed!")
                            
                            # Display results
                            st.subheader("📊 Benchmark Results")
                            
                            # Key metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                fastest_model = df.loc[df['avg_inference_time'].idxmin()]
                                st.metric(
                                    "🏃 Fastest Model", 
                                    fastest_model['model_display_name'],
                                    f"{fastest_model['avg_inference_time']:.2f}s"
                                )
                            
                            with col2:
                                best_fps = df.loc[df['fps'].idxmax()]
                                st.metric(
                                    "⚡ Best FPS",
                                    best_fps['model_display_name'], 
                                    f"{best_fps['fps']:.1f} FPS"
                                )
                            
                            with col3:
                                most_efficient = df.loc[df['avg_memory_peak_mb'].idxmin()]
                                st.metric(
                                    "💾 Most Efficient",
                                    most_efficient['model_display_name'],
                                    f"{most_efficient['avg_memory_peak_mb']:.0f}MB"
                                )
                            
                            # Detailed results table
                            st.subheader("📋 Detailed Results")
                            
                            # Format DataFrame for display
                            display_df = df[[
                                'model_display_name', 'avg_inference_time', 'fps', 
                                'avg_memory_peak_mb', 'model_load_time'
                            ]].copy()
                            
                            display_df.columns = [
                                'Model', 'Avg Time (s)', 'FPS', 'Memory (MB)', 'Load Time (s)'
                            ]
                            
                            # Round numbers for better display
                            display_df['Avg Time (s)'] = display_df['Avg Time (s)'].round(2)
                            display_df['FPS'] = display_df['FPS'].round(1)
                            display_df['Memory (MB)'] = display_df['Memory (MB)'].round(0)
                            display_df['Load Time (s)'] = display_df['Load Time (s)'].round(1)
                            
                            st.dataframe(display_df, use_container_width=True)
                            
                            # Generate and show charts
                            st.subheader("📈 Performance Visualization")
                            
                            with st.spinner("Generating charts..."):
                                benchmarker.create_comparison_charts(df)
                                benchmarker.save_results(df)
                            
                            # Display charts
                            chart_path = Path("outputs/benchmarks/model_comparison_charts.png")
                            if chart_path.exists():
                                st.image(str(chart_path), caption="Model Performance Comparison", use_container_width=True)
                            
                            # Recommendations
                            st.subheader("💡 Recommendations")
                            
                            fastest = df.loc[df['avg_inference_time'].idxmin()]
                            most_efficient = df.loc[df['avg_memory_peak_mb'].idxmin()]
                            
                            st.markdown(f"""
                            **For Real-time Applications:** Use **{fastest['model_display_name']}** 
                            - Processes images in {fastest['avg_inference_time']:.2f}s ({fastest['fps']:.1f} FPS)
                            
                            **For Memory-constrained Systems:** Use **{most_efficient['model_display_name']}**
                            - Uses only {most_efficient['avg_memory_peak_mb']:.0f}MB of memory
                            
                            **For Best Quality:** Use **Intel/dpt-large** if available
                            - Highest quality depth maps, slower processing
                            """)
                            
                            # Download results
                            st.subheader("📥 Download Results")
                            
                            csv_path = Path("outputs/benchmarks/benchmark_results.csv")
                            if csv_path.exists():
                                with open(csv_path, 'rb') as f:
                                    st.download_button(
                                        label="📊 Download Benchmark Data (CSV)",
                                        data=f.read(),
                                        file_name="depth_model_benchmark.csv",
                                        mime="text/csv"
                                    )
                        
                        else:
                            st.error("❌ Benchmark failed - no results generated")
                    
                    except Exception as e:
                        st.error(f"❌ Benchmark failed: {str(e)}")
                        st.exception(e)
    
    with tab3:
        st.header("🎥 Real-time Webcam Processing")
        st.info("Webcam processing will be available in a future update. Currently supports image upload only.")
        
        # Placeholder for webcam functionality
        st.markdown("""
        **Coming Soon:**
        - Real-time webcam depth estimation
        - Live point cloud generation
        - Video file processing
        - Export video with depth overlay
        """)
    
    with tab3:
        st.header("About This Application")
        
        st.markdown("""
        ### 🔍 Depth Estimation & Point Cloud Visualization
        
        This application demonstrates advanced computer vision techniques for:
        
        **Depth Estimation:**
        - Uses state-of-the-art MiDaS and DPT models
        - Estimates depth from single monocular images
        - Supports multiple model architectures for different speed/quality trade-offs
        
        **Point Cloud Generation:**
        - Converts 2D images + depth maps into 3D point clouds
        - Includes color information from original images
        - Supports downsampling and filtering for optimization
        
        **Key Features:**
        - Multiple pre-trained depth estimation models
        - Interactive parameter tuning
        - Export capabilities (depth maps, point clouds)
        - Optimized for both quality and performance
        
        ### 🛠️ Technical Details
        
        **Models Used:**
        - **DPT-Large**: Highest quality, uses Vision Transformer architecture
        - **DPT-Hybrid**: Balanced performance and speed
        - **MiDaS-Small**: Fastest inference, good for real-time applications
        
        **Libraries:**
        - PyTorch & Transformers for deep learning
        - Open3D for 3D point cloud processing
        - OpenCV for image processing
        - Streamlit for web interface
        
        ### 👨‍💻 About the Developer
        
        **Nikolaos Benetos**
        - AI Engineer at Accenture
        - ECE Graduate from NTUA
        - Focus: Computer Vision, Machine Learning, Robotics
        - [GitHub](https://github.com/nikolasb10) | [LinkedIn](https://linkedin.com/in/nikolasbenetos)
        
        ### 📚 Applications
        
        This technology is valuable for:
        - **Robotics**: Navigation, obstacle avoidance, SLAM
        - **Autonomous Vehicles**: Depth perception, 3D mapping
        - **AR/VR**: Scene understanding, occlusion handling
        - **3D Reconstruction**: Creating 3D models from 2D images
        - **Medical Imaging**: Volumetric analysis, surgical planning
        """)


if __name__ == "__main__":
    main()