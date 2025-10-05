#!/usr/bin/env python3
"""
Smart demo script that checks for local models first, then downloads if needed.
"""

import sys
import os
from pathlib import Path
import random

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from depth_estimator import DepthEstimator
from pointcloud_generator import PointCloudGenerator


def check_models_availability():
    """Check which models are available locally."""
    project_root = Path(__file__).parent.parent
    models_cache = project_root / "models_cache"
    
    available_models = []
    models_to_check = [
        "Intel/dpt-hybrid-midas",
        "Intel/dpt-large", 
        "Intel/dpt-beit-base-384"
    ]
    
    if models_cache.exists():
        for model_name in models_to_check:
            cache_name = model_name.replace("/", "--")
            model_path = models_cache / "models--" / cache_name
            if model_path.exists():
                available_models.append(model_name)
    
    return available_models


def download_models_if_needed():
    """Download models if none are available locally."""
    available_models = check_models_availability()
    
    if not available_models:
        print("📥 No models found locally. Downloading...")
        import subprocess
        
        script_path = Path(__file__).parent.parent / "scripts" / "download_models.py"
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Models downloaded successfully!")
            return check_models_availability()
        else:
            print(f"❌ Download failed: {result.stderr}")
            return []
    
    return available_models


def select_best_model(available_models):
    """Select the best available model."""
    # Priority order: hybrid > large > beit
    priority_order = [
        "Intel/dpt-hybrid-midas",
        "Intel/dpt-large",
        "Intel/dpt-beit-base-384"
    ]
    
    for model in priority_order:
        if model in available_models:
            return model
    
    # Fallback to first available
    return available_models[0] if available_models else "Intel/dpt-hybrid-midas"


def main():
    print("🔍 Smart Depth Estimation Demo")
    print("=" * 40)
    
    # Check model availability
    print("🔍 Checking for local models...")
    available_models = check_models_availability()
    
    if available_models:
        print(f"✅ Found {len(available_models)} local models:")
        for model in available_models:
            print(f"  • {model}")
    else:
        print("⚠️ No local models found")
        
        # Ask user if they want to download
        response = input("📥 Download models now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            available_models = download_models_if_needed()
        else:
            print("⬇️ Models will be downloaded automatically when needed")
            available_models = ["Intel/dpt-hybrid-midas"]  # Will trigger download
    
    # Select best model
    selected_model = select_best_model(available_models)
    print(f"🤖 Using model: {selected_model}")
    
    # Find sample images
    sample_dirs = [
        Path("sample_images/synthetic"),
        Path("sample_images/downloaded")
    ]
    
    # Collect all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    
    for sample_dir in sample_dirs:
        if sample_dir.exists():
            for ext in image_extensions:
                all_images.extend(sample_dir.glob(f"*{ext}"))
                all_images.extend(sample_dir.glob(f"*{ext.upper()}"))
    
    if not all_images:
        print("❌ No sample images found. Run create_sample_images.py first.")
        return 1
    
    print(f"📸 Found {len(all_images)} sample images")
    
    # Process subset of images
    demo_images = all_images[:3]  # Process first 3 images
    
    # Initialize components
    print("🚀 Initializing depth estimator...")
    depth_estimator = DepthEstimator(model_name=selected_model)
    pcg = PointCloudGenerator()
    
    # Create output directory
    output_dir = Path("outputs/smart_demo_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each demo image
    for i, img_path in enumerate(demo_images):
        print(f"\n📸 Processing {img_path.name} ({i+1}/{len(demo_images)})")
        
        try:
            # Estimate depth
            original, depth_map, depth_normalized = depth_estimator.estimate_depth_from_image(str(img_path))
            
            # Save depth visualization
            output_name = f"smart_demo_{i+1}_{img_path.stem}"
            depth_viz_path = output_dir / f"{output_name}_depth.png"
            
            fig = depth_estimator.visualize_depth(original, depth_map, str(depth_viz_path))
            
            # Generate point cloud
            h, w = depth_map.shape[:2]
            pcg.estimate_camera_intrinsics(w, h, fov_degrees=60)
            pointcloud = pcg.depth_to_pointcloud(original, depth_map, max_depth=10.0)
            
            # Downsample and filter
            if len(pointcloud.points) > 50000:
                pointcloud = pcg.downsample_pointcloud(pointcloud, voxel_size=0.05)
            pointcloud = pcg.filter_pointcloud(pointcloud)
            
            # Save point cloud
            pcd_path = output_dir / f"{output_name}_pointcloud.ply"
            pcg.save_pointcloud(pointcloud, str(pcd_path))
            
            print(f"  ✅ Depth visualization: {depth_viz_path}")
            print(f"  ✅ Point cloud ({len(pointcloud.points)} points): {pcd_path}")
            
        except Exception as e:
            print(f"  ❌ Error processing {img_path.name}: {e}")
    
    print(f"\n🎉 Smart demo completed! Results saved in: {output_dir}")
    print(f"🤖 Used model: {selected_model}")
    print(f"⚡ Models loaded from: {'local cache' if available_models else 'download'}")
    
    return 0


if __name__ == "__main__":
    exit(main())