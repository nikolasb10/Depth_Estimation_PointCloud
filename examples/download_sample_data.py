#!/usr/bin/env python3
"""
Download sample datasets for depth estimation demonstrations.
"""

import os
import sys
import kaggle
import zipfile
import shutil
from pathlib import Path


def setup_kaggle_api():
    """
    Setup Kaggle API credentials.
    Make sure you have ~/.kaggle/kaggle.json with your API credentials.
    """
    try:
        # Test if Kaggle API is properly configured
        kaggle.api.authenticate()
        print("✅ Kaggle API authenticated successfully")
        return True
    except Exception as e:
        print(f"❌ Kaggle API authentication failed: {e}")
        print("Please ensure you have:")
        print("1. Created a Kaggle account")
        print("2. Generated API token at https://www.kaggle.com/settings")
        print("3. Placed kaggle.json in ~/.kaggle/ directory")
        print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False


def download_nyu_depth_v2():
    """Download NYU Depth Dataset V2 sample images."""
    print("📥 Downloading NYU Depth V2 sample dataset...")
    
    dataset_name = "soumikrakshit/nyu-depth-v2"
    download_path = "sample_images/nyu_depth_v2"
    
    try:
        kaggle.api.dataset_download_files(
            dataset_name, 
            path=download_path, 
            unzip=True
        )
        print(f"✅ NYU Depth V2 downloaded to: {download_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download NYU Depth V2: {e}")
        return False


def download_kitti_sample():
    """Download KITTI dataset sample."""
    print("📥 Downloading KITTI sample dataset...")
    
    dataset_name = "klemenko/kitti-dataset"
    download_path = "sample_images/kitti"
    
    try:
        kaggle.api.dataset_download_files(
            dataset_name,
            path=download_path,
            unzip=True
        )
        print(f"✅ KITTI sample downloaded to: {download_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download KITTI: {e}")
        return False


def download_general_images():
    """Download general indoor/outdoor images for testing."""
    print("📥 Downloading general test images...")
    
    dataset_name = "prasunroy/natural-images"
    download_path = "sample_images/natural_images"
    
    try:
        kaggle.api.dataset_download_files(
            dataset_name,
            path=download_path,
            unzip=True
        )
        print(f"✅ Natural images downloaded to: {download_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download natural images: {e}")
        return False


def organize_sample_images():
    """Organize downloaded images into a clean structure."""
    print("📁 Organizing sample images...")
    
    base_dir = Path("sample_images")
    organized_dir = base_dir / "organized"
    organized_dir.mkdir(exist_ok=True)
    
    # Create categories
    categories = ["indoor", "outdoor", "objects", "people"]
    for category in categories:
        (organized_dir / category).mkdir(exist_ok=True)
    
    # Move and rename some sample images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Collect all images
    all_images = []
    for ext in image_extensions:
        all_images.extend(base_dir.rglob(f"*{ext}"))
        all_images.extend(base_dir.rglob(f"*{ext.upper()}"))
    
    # Copy first 20 images as samples
    for i, img_path in enumerate(all_images[:20]):
        if img_path.is_file():
            # Determine category based on filename/path
            img_name = img_path.name.lower()
            
            if any(word in img_name for word in ['indoor', 'room', 'kitchen', 'bedroom']):
                category = "indoor"
            elif any(word in img_name for word in ['outdoor', 'street', 'building', 'landscape']):
                category = "outdoor"
            elif any(word in img_name for word in ['person', 'people', 'human']):
                category = "people"
            else:
                category = "objects"
            
            # Copy to organized directory
            new_name = f"sample_{i+1:02d}{img_path.suffix}"
            new_path = organized_dir / category / new_name
            
            try:
                shutil.copy2(img_path, new_path)
                print(f"✅ Copied {img_path.name} -> {category}/{new_name}")
            except Exception as e:
                print(f"⚠️ Failed to copy {img_path.name}: {e}")
    
    print(f"📁 Sample images organized in: {organized_dir}")


def create_demo_script():
    """Create a demonstration script using the downloaded data."""
    demo_script = """#!/usr/bin/env python3
\"\"\"
Demo script using downloaded sample images.
\"\"\"

import sys
import os
from pathlib import Path
import random

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from depth_estimator import DepthEstimator
from pointcloud_generator import PointCloudGenerator


def main():
    print("🔍 Running Depth Estimation Demo with Sample Data")
    print("=" * 50)
    
    # Find sample images
    sample_dir = Path("sample_images/organized")
    if not sample_dir.exists():
        print("❌ Sample images not found. Run download_sample_data.py first.")
        return 1
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(sample_dir.rglob(f"*{ext}"))
        all_images.extend(sample_dir.rglob(f"*{ext.upper()}"))
    
    if not all_images:
        print("❌ No sample images found.")
        return 1
    
    # Select random images for demo
    demo_images = random.sample(all_images, min(5, len(all_images)))
    
    # Initialize depth estimator
    print("Loading depth estimation model...")
    depth_estimator = DepthEstimator(model_name="Intel/dpt-hybrid-midas")
    pcg = PointCloudGenerator()
    
    # Create output directory
    output_dir = Path("outputs/demo_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each demo image
    for i, img_path in enumerate(demo_images):
        print(f"\\n📸 Processing {img_path.name} ({i+1}/{len(demo_images)})")
        
        try:
            # Estimate depth
            original, depth_map, depth_normalized = depth_estimator.estimate_depth_from_image(str(img_path))
            
            # Save depth visualization
            output_name = f"demo_{i+1}_{img_path.stem}"
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
            
            print(f"  ✅ Saved depth visualization: {depth_viz_path}")
            print(f"  ✅ Saved point cloud ({len(pointcloud.points)} points): {pcd_path}")
            
        except Exception as e:
            print(f"  ❌ Error processing {img_path.name}: {e}")
    
    print(f"\\n🎉 Demo completed! Results saved in: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
"""
    
    with open("examples/run_demo_with_data.py", "w") as f:
        f.write(demo_script)
    
    # Make it executable
    os.chmod("examples/run_demo_with_data.py", 0o755)
    print("✅ Created demo script: examples/run_demo_with_data.py")


def main():
    print("🚀 Depth Estimation Dataset Downloader")
    print("=" * 40)
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Setup Kaggle API
    if not setup_kaggle_api():
        return 1
    
    # Create sample_images directory
    Path("sample_images").mkdir(exist_ok=True)
    
    # Download datasets
    success_count = 0
    
    datasets_to_try = [
        ("NYU Depth V2", download_nyu_depth_v2),
        ("KITTI Sample", download_kitti_sample),
        ("Natural Images", download_general_images),
    ]
    
    for name, download_func in datasets_to_try:
        try:
            if download_func():
                success_count += 1
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")
    
    if success_count > 0:
        print(f"\\n✅ Successfully downloaded {success_count} datasets")
        
        # Organize images
        organize_sample_images()
        
        # Create demo script
        create_demo_script()
        
        print("\\n🎯 Next steps:")
        print("1. Run: python examples/run_demo_with_data.py")
        print("2. Or start the Streamlit app: streamlit run app.py")
        print("3. Sample images are in: sample_images/organized/")
        
    else:
        print("❌ Failed to download any datasets")
        print("You can still run the demo with your own images!")
    
    return 0


if __name__ == "__main__":
    exit(main())