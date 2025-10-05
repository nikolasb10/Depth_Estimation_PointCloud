# 🔍 Depth Estimation & Point Cloud Visualization

Real-time depth estimation from monocular images with 3D point cloud generation and visualization.

![Demo](outputs/demo_preview.png)

## 🚀 Features

- **Monocular Depth Estimation**: Extract depth information from single RGB images
- **Multiple Model Support**: Choose from DPT-Large, DPT-Hybrid, and MiDaS models
- **3D Point Cloud Generation**: Convert depth maps to interactive 3D point clouds
- **Interactive Web Interface**: User-friendly Streamlit application
- **Real-time Processing**: Optimized for both quality and performance
- **Export Capabilities**: Save depth maps and point clouds for further analysis

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster inference)

### Quick Setup

1. **Clone or download this project**
   ```bash
   cd Depth_Estimation_PointCloud
   ```

2. **Install dependencies using pip**
   ```bash
   pip install torch torchvision opencv-python numpy matplotlib Pillow streamlit open3d transformers timm
   ```

   Or using the pyproject.toml:
   ```bash
   pip install -e .
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

### Alternative: Using uv (recommended)

```bash
# Install uv if you haven't already
pip install uv

# Install dependencies
uv pip install -e .

# Run the application
streamlit run app.py
```

## 📋 Usage

### Web Interface

1. **Launch the app**: `streamlit run app.py`
2. **Upload an image** or select a sample image
3. **Adjust parameters** in the sidebar:
   - Model selection (quality vs speed trade-off)
   - Point cloud settings (max depth, voxel size)
   - Camera parameters (field of view)
4. **View results**: depth maps and 3D point clouds
5. **Download outputs**: depth maps (.png) and point clouds (.ply)

### Programmatic Usage

```python
from src.depth_estimator import DepthEstimator
from src.pointcloud_generator import PointCloudGenerator

# Initialize components
depth_estimator = DepthEstimator(model_name="Intel/dpt-large")
pcg = PointCloudGenerator()

# Process an image
original, depth_map, depth_normalized = depth_estimator.estimate_depth_from_image("path/to/image.jpg")

# Generate point cloud
pointcloud = pcg.depth_to_pointcloud(original, depth_map, max_depth=10.0)

# Visualize
pcg.visualize_pointcloud(pointcloud)

# Save results
pcg.save_pointcloud(pointcloud, "output.ply")
```

### Command Line Examples

```python
# Basic depth estimation
python src/depth_estimator.py

# Point cloud generation
python src/pointcloud_generator.py

# Process video stream (webcam)
from src.depth_estimator import DepthEstimator
estimator = DepthEstimator()
estimator.process_video_stream(source=0)  # 0 for webcam
```

## 🏗️ Architecture

### Project Structure

```
Depth_Estimation_PointCloud/
├── src/
│   ├── __init__.py
│   ├── depth_estimator.py      # Depth estimation using MiDaS/DPT models
│   └── pointcloud_generator.py # 3D point cloud generation and visualization
├── examples/                   # Example scripts and notebooks
├── sample_images/             # Sample input images
├── outputs/                   # Generated outputs (depth maps, point clouds)
├── app.py                     # Streamlit web application
├── pyproject.toml            # Project dependencies and metadata
└── README.md                 # This file
```

### Key Components

1. **DepthEstimator** (`src/depth_estimator.py`)
   - Handles depth estimation using pre-trained models
   - Supports image and video processing
   - Multiple model architectures (DPT, MiDaS)

2. **PointCloudGenerator** (`src/pointcloud_generator.py`)
   - Converts depth maps to 3D point clouds
   - Camera intrinsic parameter estimation
   - Point cloud filtering and downsampling
   - Mesh generation capabilities

3. **Streamlit App** (`app.py`)
   - Interactive web interface
   - Real-time parameter adjustment
   - Download capabilities

## 🎯 Models Supported

| Model | Description | Speed | Quality | Memory |
|-------|-------------|-------|---------|---------|
| **Intel/dpt-large** | DPT with Vision Transformer backbone | Slow | Highest | High |
| **Intel/dpt-hybrid-midas** | Hybrid DPT-MiDaS architecture | Medium | High | Medium |
| **Intel/midas-v21-small** | Lightweight MiDaS model | Fast | Good | Low |

## 🔧 Configuration

### Camera Intrinsics

The system automatically estimates camera intrinsic parameters, but you can provide your own:

```python
pcg.set_camera_intrinsics(
    fx=800,      # Focal length x
    fy=800,      # Focal length y  
    cx=320,      # Principal point x
    cy=240,      # Principal point y
    width=640,   # Image width
    height=480   # Image height
)
```

### Performance Tuning

- **GPU Acceleration**: Models automatically use CUDA if available
- **Point Cloud Downsampling**: Adjust `voxel_size` for performance vs quality
- **Memory Optimization**: Use smaller models for limited memory systems
- **Batch Processing**: Process multiple images efficiently

## 📊 Technical Details

### Depth Estimation

- **Input**: RGB images (any resolution)
- **Output**: Relative depth maps (normalized to 0-255)
- **Models**: Based on MiDaS and DPT architectures
- **Preprocessing**: Automatic image normalization and resizing

### Point Cloud Generation

- **Method**: Perspective projection using camera intrinsics
- **Color**: RGB values from original image
- **Filtering**: Statistical outlier removal
- **Export**: PLY format compatible with MeshLab, CloudCompare

### Performance Metrics

On a typical setup (GTX 1080, 8GB RAM):
- **DPT-Large**: ~2-3 seconds per image (512x384)
- **DPT-Hybrid**: ~1-2 seconds per image (512x384)  
- **MiDaS-Small**: ~0.5-1 second per image (256x256)

## 🎨 Applications

This project demonstrates skills relevant to:

- **Autonomous Vehicles**: Depth perception for navigation
- **Robotics**: SLAM, obstacle avoidance, manipulation
- **AR/VR**: Scene understanding, occlusion handling
- **3D Reconstruction**: Creating 3D models from photographs
- **Medical Imaging**: Volumetric analysis from 2D scans

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Real-time webcam processing
- [ ] Video file batch processing  
- [ ] Additional depth estimation models
- [ ] Advanced point cloud algorithms (normal estimation, segmentation)
- [ ] Performance optimizations
- [ ] Mobile/edge deployment

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Nikolaos Benetos**
- AI Engineer at Accenture
- ECE Graduate from National Technical University of Athens (NTUA)
- Specialization: Computer Vision, Machine Learning, Robotics

**Contact:**
- 📧 Email: nkbenetos@gmail.com
- 💼 LinkedIn: [linkedin.com/in/nikolasbenetos](https://linkedin.com/in/nikolasbenetos)
- 🐙 GitHub: [github.com/nikolasb10](https://github.com/nikolasb10)

## 🙏 Acknowledgments

- **Intel Labs** for MiDaS and DPT models
- **HuggingFace** for model hosting and transformers library
- **Open3D** for 3D point cloud processing
- **Streamlit** for the web interface framework

## 📚 References

1. Ranftl, R., et al. "Vision Transformers for Dense Prediction." ICCV 2021.
2. Ranftl, R., et al. "Towards Robust Monocular Depth Estimation." IEEE TPAMI 2022.
3. Zhou, Q-Y., et al. "Open3D: A Modern Library for 3D Data Processing." arXiv 2018.

---

*This project showcases advanced computer vision techniques for depth estimation and 3D reconstruction, demonstrating practical applications in robotics, autonomous systems, and AR/VR.*