#!/usr/bin/env python3
"""
Create sample images for depth estimation demonstrations without needing Kaggle.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import urllib.request
import ssl


def create_synthetic_images():
    """Create synthetic test images with various depth patterns."""
    
    output_dir = Path("sample_images/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    width, height = 640, 480
    
    # 1. Circular depth pattern
    print("Creating circular depth pattern...")
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Create depth based on distance from center
    radius = np.sqrt(X**2 + Y**2)
    depth_pattern = np.exp(-radius * 2)  # Exponential falloff
    
    # Convert to RGB with depth-based coloring
    r = (np.sin(radius * 8) + 1) / 2 * depth_pattern
    g = (np.cos(radius * 6) + 1) / 2 * depth_pattern
    b = depth_pattern
    
    rgb_circular = np.stack([r, g, b], axis=-1)
    rgb_circular = (rgb_circular * 255).astype(np.uint8)
    
    Image.fromarray(rgb_circular).save(output_dir / "circular_pattern.png")
    
    # 2. Geometric shapes
    print("Creating geometric shapes...")
    img = Image.new('RGB', (width, height), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw rectangles at different "depths" (different colors)
    draw.rectangle([50, 50, 200, 150], fill='red')      # Close object
    draw.rectangle([250, 100, 400, 200], fill='green')  # Medium distance
    draw.rectangle([450, 150, 590, 250], fill='blue')   # Far object
    
    # Draw circles
    draw.ellipse([100, 250, 200, 350], fill='yellow')   # Close
    draw.ellipse([300, 300, 450, 450], fill='purple')   # Far
    
    img.save(output_dir / "geometric_shapes.png")
    
    # 3. Gradient patterns
    print("Creating gradient patterns...")
    
    # Horizontal gradient
    gradient_h = np.linspace(0, 1, width)
    gradient_h = np.tile(gradient_h, (height, 1))
    
    # Vertical gradient  
    gradient_v = np.linspace(0, 1, height)
    gradient_v = np.tile(gradient_v.reshape(-1, 1), (1, width))
    
    # Combine gradients
    combined = gradient_h * 0.7 + gradient_v * 0.3
    
    # Convert to RGB
    rgb_gradient = np.stack([combined, 1-combined, combined*0.5], axis=-1)
    rgb_gradient = (rgb_gradient * 255).astype(np.uint8)
    
    Image.fromarray(rgb_gradient).save(output_dir / "gradient_pattern.png")
    
    # 4. Checkerboard pattern
    print("Creating checkerboard pattern...")
    checker_size = 40
    checkerboard = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(0, height, checker_size):
        for j in range(0, width, checker_size):
            if (i // checker_size + j // checker_size) % 2 == 0:
                # White squares (closer)
                checkerboard[i:i+checker_size, j:j+checker_size] = [255, 255, 255]
            else:
                # Colored squares (farther)
                color = [
                    int(255 * (i / height)),      # Red varies with height
                    int(255 * (j / width)),       # Green varies with width  
                    128                           # Blue constant
                ]
                checkerboard[i:i+checker_size, j:j+checker_size] = color
    
    Image.fromarray(checkerboard).save(output_dir / "checkerboard_pattern.png")
    
    print(f"✅ Created 4 synthetic images in: {output_dir}")
    return output_dir


def download_free_images():
    """Download some free sample images from the internet."""
    
    output_dir = Path("sample_images/downloaded")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Free sample images (no attribution required)
    sample_urls = {
        "indoor_room.jpg": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=640&h=480&fit=crop",
        "outdoor_landscape.jpg": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640&h=480&fit=crop",
        "street_scene.jpg": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640&h=480&fit=crop",
        "building.jpg": "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=640&h=480&fit=crop",
        "nature.jpg": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=640&h=480&fit=crop"
    }
    
    # Create SSL context that doesn't verify certificates (for demo purposes)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    downloaded_count = 0
    
    for filename, url in sample_urls.items():
        try:
            print(f"Downloading {filename}...")
            output_path = output_dir / filename
            
            request = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            with urllib.request.urlopen(request, context=ssl_context) as response:
                with open(output_path, 'wb') as f:
                    f.write(response.read())
            
            print(f"✅ Downloaded: {filename}")
            downloaded_count += 1
            
        except Exception as e:
            print(f"⚠️ Failed to download {filename}: {e}")
    
    if downloaded_count > 0:
        print(f"✅ Downloaded {downloaded_count} sample images to: {output_dir}")
    else:
        print("❌ No images could be downloaded")
    
    return output_dir


def create_demo_script():
    """Create a demo script that uses the sample images."""
    
    demo_script = '''#!/usr/bin/env python3
"""
Demo script using sample images (no Kaggle required).
"""

import sys
import os
from pathlib import Path
import random

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from depth_estimator import DepthEstimator
from pointcloud_generator import PointCloudGenerator


def main():
    print("🔍 Running Depth Estimation Demo with Sample Images")
    print("=" * 50)
    
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
    
    # Process all images
    demo_images = all_images[:5]  # Process first 5 images
    
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
            
            print(f"  ✅ Depth visualization: {depth_viz_path}")
            print(f"  ✅ Point cloud ({len(pointcloud.points)} points): {pcd_path}")
            
        except Exception as e:
            print(f"  ❌ Error processing {img_path.name}: {e}")
    
    print(f"\\n🎉 Demo completed! Results saved in: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())'''
    
    with open("examples/run_demo.py", "w") as f:
        f.write(demo_script)
    
    print("✅ Created demo script: examples/run_demo.py")


def main():
    print("🖼️ Creating Sample Images for Depth Estimation Demo")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Create synthetic images
    synthetic_dir = create_synthetic_images()
    
    # Try to download real images
    print("\nDownloading real sample images...")
    downloaded_dir = download_free_images()
    
    # Create demo script
    create_demo_script()
    
    print("\n🎯 Sample images created successfully!")
    print(f"Synthetic images: {synthetic_dir}")
    print(f"Downloaded images: {downloaded_dir}")
    print("\n🚀 Next steps:")
    print("1. Run: python examples/run_demo.py")
    print("2. Or start Streamlit app: streamlit run app.py")
    
    return 0


if __name__ == "__main__":
    exit(main())