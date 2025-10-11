import streamlit as st

def render_about():
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