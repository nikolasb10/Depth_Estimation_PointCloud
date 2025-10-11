"""
Streamlit component for Video Depth Reconstruction functionality.
Extracted from main app.py for better code organization.
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

from video_depth_processor import VideoDepthProcessor, CameraIntrinsics


@st.cache_resource
def init_video_processor(model_name: str):
    """Initialize video processor with caching."""
    try:
        processor = VideoDepthProcessor(model_name=model_name)
        return processor
    except Exception as e:
        st.error(f"Failed to initialize video processor: {str(e)}")
        return None


def check_sample_videos():
    """Check if sample videos are available."""
    sample_dir = Path("sample_videos")
    return sample_dir.exists() and any(sample_dir.glob("*.mp4"))


def display_frame_sequence(results: list, max_frames: int = 5):
    """Display a sequence of processed frames."""
    st.subheader("📽️ Frame Processing Sequence")
    
    # Limit frames for display
    display_results = results[:max_frames] if len(results) > max_frames else results
    
    cols = st.columns(len(display_results))
    
    for i, (col, result) in enumerate(zip(cols, display_results)):
        with col:
            frame_idx = result['frame_idx']
            st.write(f"**Frame {frame_idx}**")
            
            # RGB frame thumbnail
            rgb_small = cv2.resize(result['rgb_frame'], (150, 100))
            st.image(rgb_small, caption=f"RGB Frame {frame_idx}")
            
            # Point count
            point_count = len(result['pointcloud'].points)
            st.metric("Points", f"{point_count:,}")
    
    if len(results) > max_frames:
        st.info(f"Showing first {max_frames} of {len(results)} processed frames")


def create_3d_plot(pointcloud):
    """Create interactive 3D plot of point cloud using Plotly."""
    try:
        import open3d as o3d
        
        points = np.asarray(pointcloud.points)
        if len(points) == 0:
            return None
        
        # Sample points for performance if too many
        max_points = 10000
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
            title="3D Point Cloud Visualization",
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


def render_video_depth_reconstruction():
    """Main function to render the Video Depth Reconstruction tab."""
    st.header("🎥 Video Depth Reconstruction")
    st.markdown("""
    Transform videos into dense 3D reconstructions using monocular depth estimation and camera motion tracking.
    """)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; text-align: center;">
            <h4>🔍 STEP 1</h4>
            <p>Depth → 3D Point Cloud<br/>
            Convert per-pixel depth to 3D coordinates using camera intrinsics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; text-align: center;">
            <h4>📹 STEP 2</h4>
            <p>Video → Dense 3D Scene<br/>
            Track camera motion and fuse multiple depth maps</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; text-align: center;">
            <h4>🎯 Visual SLAM</h4>
            <p>Feature-based Motion Estimation<br/>
            ORB/SIFT features + Essential Matrix decomposition</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Get selected model from session state (passed from main app)
    selected_model = st.session_state.get('selected_model', 'Intel/dpt-hybrid-midas')
    
    # Sidebar configuration for video processing
    with st.sidebar:
        st.header("⚙️ Video Processing Configuration")
        
        # Processing parameters
        st.subheader("Processing Parameters")
        
        video_max_frames = st.slider("Max Frames to Process", 10, 200, 50)
        video_frame_skip = st.slider("Frame Skip (process every N frames)", 1, 10, 2)
        video_max_depth = st.slider("Max Depth (meters)", 1.0, 20.0, 10.0)
        video_voxel_size = st.slider("Voxel Size for Merging", 0.01, 0.2, 0.05)
        
        # Camera intrinsics
        st.subheader("Camera Parameters")
        
        use_default_intrinsics = st.checkbox("Use Default Webcam Intrinsics", True)
        
        if not use_default_intrinsics:
            focal_length = st.number_input("Focal Length (pixels)", 100.0, 2000.0, 500.0)
            video_fov_degrees = st.slider("Field of View (degrees)", 30, 120, 60)
        
        # Initialize video processor
        if st.button("🔄 Load Video Model", type="primary"):
            with st.spinner("Loading video model..."):
                processor = init_video_processor(selected_model)
                if processor:
                    st.session_state.video_processor = processor
                    st.session_state.video_model_loaded = True
                    st.success("✅ Video model loaded successfully!")
                    st.rerun()  # Force rerun to update the UI
                else:
                    st.session_state.video_processor = None
                    st.session_state.video_model_loaded = False
                    st.error("❌ Failed to load video model")
    
    # Main content - just the video processing (tabs are handled in main app.py)
    render_video_processing_tab(
        video_max_frames, video_frame_skip, video_max_depth, 
        video_voxel_size, use_default_intrinsics,
        video_fov_degrees if not use_default_intrinsics else 60
    )


def render_video_processing_tab(video_max_frames, video_frame_skip, video_max_depth, 
                               video_voxel_size, use_default_intrinsics, video_fov_degrees):
    """Render the video processing tab."""
    st.header("Video Processing Pipeline")
    
    # Video input section
    st.subheader("1️⃣ Video Input")
    
    video_source = st.radio(
        "Choose video source:",
        ["Upload Video", "Use Sample Videos"],
        horizontal=True
    )
    
    video_file = None
    video_path = None
    
    if video_source == "Upload Video":
        video_file = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a short video (< 30 seconds recommended for demo)"
        )
        
        if video_file:
            # Save uploaded video temporarily
            import tempfile  # Ensure tempfile is available in local scope
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_file.read())
                video_path = tmp_file.name
    
    else:
        # Sample video selection
        st.info("🎬 Choose from available sample videos for testing depth estimation!")
        
        sample_dir = Path("sample_videos")
        if sample_dir.exists():
            sample_videos = list(sample_dir.glob("*.mp4"))
            if sample_videos:
                selected_video = st.selectbox(
                    "Choose sample video:",
                    sample_videos,
                    format_func=lambda x: x.name
                )
                video_path = str(selected_video)
                # Show metadata if available
                metadata_path = Path(str(selected_video) + ".txt")
                if metadata_path.exists():
                    with st.expander("📋 Video Metadata"):
                        with open(metadata_path, 'r') as f:
                            st.text(f.read())
            else:
                st.warning("No sample videos found in sample_videos/ directory")
        else:
            st.warning("Sample videos directory not found")
    
    # Video preview
    if video_path and os.path.exists(video_path):
        st.subheader("2️⃣ Video Preview")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            try:
                st.video(video_path)
                st.success("✅ Video loaded successfully")
            except Exception as e:
                st.error(f"❌ Video display failed: {str(e)}")
                # Fallback: show first frame as preview
                try:
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="First frame preview", use_column_width=True)
                except Exception as frame_e:
                    st.error(f"Frame preview also failed: {str(frame_e)}")
        
        with col2:
            # Video info
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                
                st.metric("Resolution", f"{width}x{height}")
                st.metric("FPS", fps)
                st.metric("Duration", f"{duration:.1f}s")
                st.metric("Total Frames", frame_count)
                
                cap.release()
    
    # Processing section
    if video_path and os.path.exists(video_path):
        st.subheader("3️⃣ Process Video")
        
        # Check if video processor is loaded
        video_processor_loaded = (
            st.session_state.get('video_processor') is not None and 
            st.session_state.get('video_model_loaded', False)
        )
        
        if not video_processor_loaded:
            st.warning("⚠️ Please load a video model first using the sidebar.")
        else:
            # Set processing parameters
            st.session_state.video_processor.max_frames = video_max_frames
            st.session_state.video_processor.frame_skip = video_frame_skip
            st.session_state.video_processor.max_depth = video_max_depth
            
            # Get video dimensions for camera intrinsics
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Camera intrinsics
            if use_default_intrinsics:
                # Check if this is a KITTI video for better intrinsics
                if "kitti" in str(video_path).lower():
                    # Use KITTI camera intrinsics
                    camera_intrinsics = CameraIntrinsics(
                        fx=718.856, fy=718.856, 
                        cx=607.1928, cy=185.2157,
                        width=width, height=height
                    )
                    st.info("🎯 Using KITTI camera intrinsics for better accuracy")
                else:
                    camera_intrinsics = None  # Will use default
            else:
                camera_intrinsics = CameraIntrinsics.from_fov(width, height, video_fov_degrees)
            
            if st.button("🚀 Start Video Processing", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    start_time = time.time()
                    
                    # Process video
                    status_text.text("Processing video frames...")
                    frame_results = st.session_state.video_processor.process_video_to_pointclouds(
                        video_path, camera_intrinsics
                    )
                    progress_bar.progress(0.7)
                    
                    # Merge point clouds
                    status_text.text("Merging point clouds into 3D scene...")
                    merged_scene = st.session_state.video_processor.merge_pointclouds(
                        frame_results, video_voxel_size
                    )
                    progress_bar.progress(1.0)
                    
                    processing_time = time.time() - start_time
                    
                    # Store results
                    st.session_state.processing_results = {
                        'frame_results': frame_results,
                        'merged_scene': merged_scene,
                        'processing_time': processing_time,
                        'video_path': video_path
                    }
                    
                    status_text.text("✅ Processing completed!")
                    st.success(f"Successfully processed {len(frame_results)} frames in {processing_time:.1f} seconds")
                    
                    # Show quick preview
                    if len(merged_scene.points) > 0:
                        st.metric("Final Point Cloud Size", f"{len(merged_scene.points):,} points")
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()



def render_technical_details_tab():
    """Render the technical details tab."""
    st.header("ℹ️ Technical Implementation Details")
    
    st.markdown("""
    ### 🔍 STEP 1: Depth → 3D Point Cloud
    
    **Camera Intrinsics & 3D Projection:**
    """)
    
    st.code("""
# Convert depth map to 3D coordinates
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy  
Z = depth(u, v)

# Where:
# (u, v) = pixel coordinates
# (cx, cy) = principal point (image center)
# (fx, fy) = focal lengths
# Z = depth value at pixel (u, v)
    """, language="python")
    
    st.markdown("""
    ### 📹 STEP 2: Video → Dense 3D Scene
    
    **Camera Motion Estimation Pipeline:**
    1. **Feature Detection**: ORB/SIFT keypoints in consecutive frames
    2. **Feature Matching**: Find correspondences between frames  
    3. **Essential Matrix**: Estimate camera motion from matches
    4. **Pose Recovery**: Decompose essential matrix into R, t
    5. **Point Cloud Transformation**: Apply camera poses to align frames
    6. **Fusion**: Merge overlapping point clouds with voxel downsampling
    """)
    
    st.code("""
# Motion estimation workflow
kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)

matches = matcher.match(desc1, desc2)
pts1, pts2 = extract_matched_points(matches, kp1, kp2)

E, mask = cv2.findEssentialMat(pts1, pts2, camera_K)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, camera_K)

# Transform point cloud by estimated camera motion
pointcloud.transform(transformation_matrix)
    """, language="python")
    
    st.markdown("""
    ### 🎯 Key Algorithms Used
    
    | Component | Algorithm | Purpose |
    |-----------|-----------|---------|
    | **Depth Estimation** | MiDaS/DPT Transformer | Per-pixel depth from single image |
    | **Feature Detection** | ORB/SIFT | Find keypoints for motion tracking |
    | **Motion Estimation** | Essential Matrix + RANSAC | Robust camera pose estimation |
    | **3D Reconstruction** | Camera intrinsics projection | Convert depth → 3D coordinates |
    | **Point Cloud Fusion** | Voxel downsampling + outlier removal | Merge multiple point clouds |
    
    ### 🚗 KITTI Dataset Integration
    
    **Real-World Testing Data:**
    - **KITTI Dataset**: Industry-standard autonomous driving sequences
    - **Ground Truth Available**: Camera poses and calibration parameters  
    - **Challenging Scenarios**: Urban driving, varying lighting, dynamic objects
    - **Realistic Camera Motion**: Actual vehicle movement patterns
    
    ### 🔬 This Demonstrates
    
    ✅ **Camera Geometry**: Understanding intrinsic/extrinsic parameters  
    ✅ **3D Spatial Reasoning**: Depth-to-3D conversion and coordinate systems  
    ✅ **Visual SLAM Concepts**: Feature-based motion estimation  
    ✅ **Multi-view Geometry**: Essential matrix and pose recovery  
    ✅ **Point Cloud Processing**: Merging, downsampling, filtering  
    ✅ **Computer Vision Pipeline**: End-to-end video → 3D reconstruction  
    ✅ **Real-World Data Handling**: KITTI dataset processing and integration  
    ✅ **Dataset Management**: HuggingFace datasets integration  
    """)