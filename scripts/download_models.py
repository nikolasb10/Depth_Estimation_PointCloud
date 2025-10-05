#!/usr/bin/env python3
"""
Download and cache all depth estimation models locally.
Run this once to avoid waiting for model downloads during demos.
"""

import os
import sys
from pathlib import Path
from transformers import pipeline, DPTForDepthEstimation, DPTImageProcessor
import torch


def setup_cache_directory():
    """Setup local models cache directory."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models_cache"
    models_dir.mkdir(exist_ok=True)
    
    # Set environment variable for transformers cache
    os.environ['TRANSFORMERS_CACHE'] = str(models_dir)
    os.environ['HF_HOME'] = str(models_dir)
    
    print(f"📁 Models will be cached in: {models_dir}")
    return models_dir


def download_model(model_name, models_dir):
    """Download a specific model and cache it locally."""
    print(f"\n🔄 Downloading {model_name}...")
    
    try:
        # Method 1: Using pipeline (easiest)
        pipeline_model = pipeline(
            "depth-estimation",
            model=model_name,
            cache_dir=str(models_dir),
            device=-1  # CPU to avoid CUDA issues during download
        )
        
        # Method 2: Download model and processor separately (more control)
        model = DPTForDepthEstimation.from_pretrained(
            model_name, 
            cache_dir=str(models_dir)
        )
        processor = DPTImageProcessor.from_pretrained(
            model_name,
            cache_dir=str(models_dir)
        )
        
        print(f"✅ Successfully downloaded {model_name}")
        
        # Get model size info
        model_path = models_dir / "models--" / model_name.replace("/", "--")
        if model_path.exists():
            size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024*1024)
            print(f"   📊 Model size: {size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False


def verify_models(models_dir):
    """Verify that downloaded models can be loaded."""
    print(f"\n🔍 Verifying downloaded models...")
    
    models_to_verify = [
        "Intel/dpt-large",
        "Intel/dpt-hybrid-midas", 
        "Intel/dpt-beit-base-384"
    ]
    
    verified_count = 0
    
    for model_name in models_to_verify:
        try:
            # Try to load the model from cache
            pipeline_model = pipeline(
                "depth-estimation",
                model=model_name,
                cache_dir=str(models_dir),
                device=-1
            )
            print(f"✅ {model_name}: Ready to use")
            verified_count += 1
            
        except Exception as e:
            print(f"❌ {model_name}: Verification failed - {e}")
    
    return verified_count


def update_depth_estimator():
    """Update depth_estimator.py to use local cache."""
    
    depth_estimator_path = Path(__file__).parent.parent / "src" / "depth_estimator.py"
    
    if not depth_estimator_path.exists():
        print("⚠️  depth_estimator.py not found, skipping update")
        return
    
    # Read current content
    with open(depth_estimator_path, 'r') as f:
        content = f.read()
    
    # Add cache directory setup at the top
    cache_setup = '''import os
from pathlib import Path

# Setup local models cache
_project_root = Path(__file__).parent.parent
_models_cache = _project_root / "models_cache"
if _models_cache.exists():
    os.environ['TRANSFORMERS_CACHE'] = str(_models_cache)
    os.environ['HF_HOME'] = str(_models_cache)

'''
    
    # Check if already updated
    if "TRANSFORMERS_CACHE" in content:
        print("✅ depth_estimator.py already configured for local cache")
        return
    
    # Insert cache setup after existing imports
    import_end = content.find('import warnings')
    if import_end != -1:
        # Find end of import section
        next_line = content.find('\n', import_end)
        if next_line != -1:
            new_content = content[:next_line+1] + '\n' + cache_setup + content[next_line+1:]
            
            # Write updated content
            with open(depth_estimator_path, 'w') as f:
                f.write(new_content)
            
            print("✅ Updated depth_estimator.py to use local cache")
        else:
            print("⚠️  Could not update depth_estimator.py automatically")
    else:
        print("⚠️  Could not find import section in depth_estimator.py")


def main():
    print("🚀 Depth Estimation Models Downloader")
    print("=" * 50)
    
    # Setup cache directory
    models_dir = setup_cache_directory()
    
    # Models to download (ordered by priority/usefulness)
    models_to_download = [
        ("Intel/dpt-hybrid-midas", "Best balance of speed and quality"),
        ("Intel/dpt-beit-base-384", "Good alternative, faster than large"),
        ("Intel/dpt-large", "Highest quality, slower")
    ]
    
    print(f"\n📥 Will download {len(models_to_download)} models:")
    for model_name, description in models_to_download:
        print(f"  • {model_name}: {description}")
    
    # Download each model
    successful_downloads = 0
    total_models = len(models_to_download)
    
    for i, (model_name, description) in enumerate(models_to_download, 1):
        print(f"\n[{i}/{total_models}] {description}")
        if download_model(model_name, models_dir):
            successful_downloads += 1
    
    # Verify downloads
    verified_models = verify_models(models_dir)
    
    # Update depth_estimator.py to use cache
    update_depth_estimator()
    
    # Summary
    print(f"\n🎉 Download Summary:")
    print(f"  • Downloaded: {successful_downloads}/{total_models} models")
    print(f"  • Verified: {verified_models}/{total_models} models")
    print(f"  • Cache location: {models_dir}")
    
    if successful_downloads == total_models:
        print(f"\n✅ All models downloaded successfully!")
        print(f"🚀 Your Streamlit app will now load much faster!")
        print(f"\nNext steps:")
        print(f"  1. Run: streamlit run app.py")
        print(f"  2. Models will load in ~2-3 seconds instead of minutes")
    else:
        print(f"\n⚠️  Some models failed to download")
        print(f"Check your internet connection and try again")
    
    # Show disk usage
    try:
        total_size = sum(f.stat().st_size for f in models_dir.rglob('*') if f.is_file())
        print(f"\n💾 Total disk usage: {total_size / (1024*1024*1024):.1f} GB")
    except Exception:
        pass
    
    return 0 if successful_downloads == total_models else 1


if __name__ == "__main__":
    exit(main())