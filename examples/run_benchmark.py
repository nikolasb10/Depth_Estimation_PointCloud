#!/usr/bin/env python3
"""
Standalone script to run model benchmarks from command line.
"""

import sys
import os
from pathlib import Path
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_benchmarker import ModelBenchmarker


def main():
    parser = argparse.ArgumentParser(description='Benchmark depth estimation models')
    parser.add_argument('--models', nargs='+', 
                       default=['Intel/dpt-hybrid-midas', 'Intel/dpt-large'],
                       help='Models to benchmark')
    parser.add_argument('--images', type=int, default=3,
                       help='Number of test images to use')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to run benchmark on')
    parser.add_argument('--output', default='outputs/benchmarks',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🔍 Depth Estimation Model Benchmark")
    print("=" * 50)
    
    # Find test images
    sample_dirs = [
        Path("sample_images/synthetic"),
        Path("sample_images/downloaded")
    ]
    
    test_images = []
    for sample_dir in sample_dirs:
        if sample_dir.exists():
            test_images.extend(list(sample_dir.glob("*.jpg")))
            test_images.extend(list(sample_dir.glob("*.png")))
    
    if not test_images:
        print("❌ No test images found. Run create_sample_images.py first.")
        return 1
    
    # Limit test images
    test_images = [str(img) for img in test_images[:args.images]]
    
    print(f"📸 Test images: {len(test_images)}")
    print(f"🤖 Models: {args.models}")
    print(f"💻 Device: {args.device}")
    
    # Run benchmark
    benchmarker = ModelBenchmarker()
    
    device = None if args.device == 'auto' else args.device
    df = benchmarker.compare_models(test_images, args.models, device)
    
    if not df.empty:
        print(f"\n📊 Benchmark Results:")
        print("=" * 80)
        print(df[['model_display_name', 'avg_inference_time', 'fps', 'avg_memory_peak_mb']].to_string(index=False))
        
        # Create visualizations and save results
        benchmarker.create_comparison_charts(df, args.output)
        benchmarker.save_results(df, args.output)
        
        print(f"\n🎉 Benchmark completed!")
        print(f"📁 Results saved to: {args.output}")
        
    else:
        print("❌ Benchmark failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())