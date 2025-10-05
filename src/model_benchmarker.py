"""
Model comparison and benchmarking utilities for depth estimation models.
"""

import time
import psutil
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
import gc

from depth_estimator import DepthEstimator


class ModelBenchmarker:
    """
    Benchmark and compare different depth estimation models.
    """
    
    def __init__(self):
        self.results = []
        self.models = {
            "Intel/dpt-large": "DPT-Large (Highest Quality)",
            "Intel/dpt-hybrid-midas": "DPT-Hybrid (Balanced)", 
            "Intel/dpt-beit-base-384": "DPT-BEiT (Alternative)"
        }
    
    def benchmark_model(self, model_name, test_images, device=None):
        """
        Benchmark a single model on test images.
        
        Args:
            model_name (str): Model identifier
            test_images (list): List of image paths to test
            device (str): Device to run on
            
        Returns:
            dict: Benchmark results
        """
        print(f"\n🔄 Benchmarking {model_name}...")
        
        # Initialize model
        start_memory = self._get_memory_usage()
        model_load_start = time.time()
        
        try:
            estimator = DepthEstimator(model_name=model_name, device=device)
            model_load_time = time.time() - model_load_start
            model_memory = self._get_memory_usage() - start_memory
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return None
        
        # Benchmark on test images
        inference_times = []
        memory_peaks = []
        
        for i, img_path in enumerate(test_images):
            print(f"  Processing image {i+1}/{len(test_images)}: {Path(img_path).name}")
            
            # Measure inference time and memory
            start_memory = self._get_memory_usage()
            start_time = time.time()
            
            try:
                _, depth_map, _ = estimator.estimate_depth_from_image(img_path)
                
                inference_time = time.time() - start_time
                peak_memory = self._get_memory_usage() - start_memory
                
                inference_times.append(inference_time)
                memory_peaks.append(peak_memory)
                
                # Clean up to avoid memory accumulation
                del depth_map
                gc.collect()
                
            except Exception as e:
                print(f"    ❌ Error processing {Path(img_path).name}: {e}")
                continue
        
        # Calculate statistics
        if inference_times:
            results = {
                'model_name': model_name,
                'model_display_name': self.models.get(model_name, model_name),
                'model_load_time': model_load_time,
                'model_memory_mb': model_memory,
                'avg_inference_time': np.mean(inference_times),
                'min_inference_time': np.min(inference_times),
                'max_inference_time': np.max(inference_times),
                'std_inference_time': np.std(inference_times),
                'avg_memory_peak_mb': np.mean(memory_peaks),
                'max_memory_peak_mb': np.max(memory_peaks),
                'images_processed': len(inference_times),
                'fps': 1.0 / np.mean(inference_times) if inference_times else 0,
                'device': device or 'cpu'
            }
            
            print(f"  ✅ Results: {results['avg_inference_time']:.2f}s avg, {results['fps']:.1f} FPS")
            return results
        else:
            print(f"  ❌ No successful inferences for {model_name}")
            return None
    
    def compare_models(self, test_images, models=None, device=None):
        """
        Compare multiple models on the same test images.
        
        Args:
            test_images (list): List of image paths
            models (list): List of model names to compare
            device (str): Device to run on
            
        Returns:
            pd.DataFrame: Comparison results
        """
        if models is None:
            models = list(self.models.keys())
        
        print(f"🏁 Starting model comparison on {len(test_images)} images")
        print(f"Models to compare: {models}")
        
        self.results = []
        
        for model_name in models:
            result = self.benchmark_model(model_name, test_images, device)
            if result:
                self.results.append(result)
        
        # Create comparison DataFrame
        if self.results:
            df = pd.DataFrame(self.results)
            return df
        else:
            print("❌ No successful benchmarks")
            return pd.DataFrame()
    
    def create_comparison_charts(self, df, output_dir="outputs/benchmarks"):
        """
        Create visualization charts for model comparison.
        
        Args:
            df (pd.DataFrame): Benchmark results
            output_dir (str): Directory to save charts
        """
        if df.empty:
            print("No data to visualize")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        
        # 1. Inference Time Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Depth Estimation Model Comparison', fontsize=16, fontweight='bold')
        
        # Average Inference Time
        axes[0, 0].bar(df['model_display_name'], df['avg_inference_time'], 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 0].set_title('Average Inference Time')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(df['avg_inference_time']):
            axes[0, 0].text(i, v + 0.1, f'{v:.2f}s', ha='center', va='bottom')
        
        # FPS Comparison
        axes[0, 1].bar(df['model_display_name'], df['fps'], 
                       color=['#d62728', '#9467bd', '#8c564b'])
        axes[0, 1].set_title('Frames Per Second (FPS)')
        axes[0, 1].set_ylabel('FPS')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(df['fps']):
            axes[0, 1].text(i, v + 0.05, f'{v:.1f}', ha='center', va='bottom')
        
        # Memory Usage
        axes[1, 0].bar(df['model_display_name'], df['avg_memory_peak_mb'], 
                       color=['#17becf', '#bcbd22', '#e377c2'])
        axes[1, 0].set_title('Average Memory Usage')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(df['avg_memory_peak_mb']):
            axes[1, 0].text(i, v + 5, f'{v:.0f}MB', ha='center', va='bottom')
        
        # Model Load Time
        axes[1, 1].bar(df['model_display_name'], df['model_load_time'], 
                       color=['#ff9896', '#98df8a', '#c5b0d5'])
        axes[1, 1].set_title('Model Load Time')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(df['model_load_time']):
            axes[1, 1].text(i, v + 0.5, f'{v:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        chart_path = output_path / "model_comparison_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison charts saved to: {chart_path}")
        
        # 2. Performance vs Quality scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create a simple quality metric (inverse of inference time for now)
        quality_score = 1 / df['avg_inference_time']  # Higher is better
        
        # Create colors that match the number of data points
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        point_colors = [colors[i % len(colors)] for i in range(len(df))]
        
        scatter = ax.scatter(df['avg_inference_time'], quality_score, 
                           s=df['avg_memory_peak_mb']*2, alpha=0.7,
                           c=point_colors)
        
        # Add model labels
        for i, model in enumerate(df['model_display_name']):
            ax.annotate(model, (df['avg_inference_time'].iloc[i], quality_score.iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Inference Time (seconds)')
        ax.set_ylabel('Performance Score (1/time)')
        ax.set_title('Performance vs Speed Trade-off\n(Bubble size = Memory Usage)')
        
        plt.tight_layout()
        scatter_path = output_path / "performance_tradeoff.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance trade-off chart saved to: {scatter_path}")
    
    def save_results(self, df, output_dir="outputs/benchmarks"):
        """Save benchmark results to CSV."""
        if df.empty:
            print("No results to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        csv_path = output_path / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Create summary report
        summary_path = output_path / "benchmark_summary.md"
        with open(summary_path, 'w') as f:
            f.write("# Depth Estimation Model Benchmark Results\n\n")
            f.write(f"**Test Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Models Tested:** {len(df)}\n")
            f.write(f"**Images Processed:** {df['images_processed'].iloc[0] if not df.empty else 0}\n\n")
            
            f.write("## Performance Summary\n\n")
            f.write("| Model | Avg Time (s) | FPS | Memory (MB) | Load Time (s) |\n")
            f.write("|-------|--------------|-----|-------------|---------------|\n")
            
            for _, row in df.iterrows():
                f.write(f"| {row['model_display_name']} | {row['avg_inference_time']:.2f} | "
                       f"{row['fps']:.1f} | {row['avg_memory_peak_mb']:.0f} | "
                       f"{row['model_load_time']:.1f} |\n")
            
            f.write("\n## Recommendations\n\n")
            
            # Find best performers
            fastest = df.loc[df['avg_inference_time'].idxmin()]
            most_efficient = df.loc[df['avg_memory_peak_mb'].idxmin()]
            
            f.write(f"- **Fastest Model:** {fastest['model_display_name']} ({fastest['avg_inference_time']:.2f}s, {fastest['fps']:.1f} FPS)\n")
            f.write(f"- **Most Memory Efficient:** {most_efficient['model_display_name']} ({most_efficient['avg_memory_peak_mb']:.0f}MB)\n")
            f.write(f"- **Best for Real-time:** {fastest['model_display_name']} (>{fastest['fps']:.0f} FPS)\n")
            f.write(f"- **Best for Quality:** Intel/dpt-large (if available)\n")
        
        print(f"📄 Results saved:")
        print(f"  • CSV: {csv_path}")
        print(f"  • Summary: {summary_path}")
    
    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


def main():
    """Example usage of ModelBenchmarker."""
    
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
    
    # Limit to first 3 images for demo
    test_images = [str(img) for img in test_images[:3]]
    
    print(f"🔍 Running benchmark on {len(test_images)} images:")
    for img in test_images:
        print(f"  • {Path(img).name}")
    
    # Run benchmark
    benchmarker = ModelBenchmarker()
    df = benchmarker.compare_models(test_images)
    
    if not df.empty:
        print(f"\n📊 Benchmark Results:")
        print(df[['model_display_name', 'avg_inference_time', 'fps', 'avg_memory_peak_mb']].to_string(index=False))
        
        # Create visualizations
        benchmarker.create_comparison_charts(df)
        benchmarker.save_results(df)
        
        print(f"\n🎉 Benchmark completed! Check outputs/benchmarks/ for detailed results.")
    
    return 0


if __name__ == "__main__":
    exit(main())