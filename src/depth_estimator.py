"""
Depth estimation module using MiDaS model for monocular depth prediction.
"""

import os
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup local models cache
_project_root = Path(__file__).parent.parent
_models_cache = _project_root / "models_cache"
if _models_cache.exists():
    os.environ['TRANSFORMERS_CACHE'] = str(_models_cache)
    os.environ['HF_HOME'] = str(_models_cache)


class DepthEstimator:
    """
    Depth estimation using MiDaS model from transformers library.
    Supports both image and video processing.
    """
    
    def __init__(self, model_name="Intel/dpt-large", device=None):
        """
        Initialize the depth estimator.
        
        Args:
            model_name (str): HuggingFace model name for depth estimation
            device (str): Device to run inference on ('cpu', 'cuda', 'auto')
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        print(f"Loading depth estimation model: {model_name}")
        print(f"Using device: {device}")
        
        # Check if model exists locally first
        local_model_path = self._get_local_model_path(model_name)
        
        try:
            # Try to load from local cache first
            if local_model_path and local_model_path.exists():
                print("✅ Loading from local cache...")
                self.depth_estimator = pipeline(
                    "depth-estimation",
                    model=model_name,
                    cache_dir=str(_models_cache) if _models_cache.exists() else None,
                    device=0 if device == "cuda" else -1
                )
            else:
                print("⬇️ Model not found locally, downloading...")
                self.depth_estimator = pipeline(
                    "depth-estimation",
                    model=model_name,
                    cache_dir=str(_models_cache) if _models_cache.exists() else None,
                    device=0 if device == "cuda" else -1
                )
                print("✅ Model downloaded and cached for future use")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Retrying with default settings...")
            self.depth_estimator = pipeline(
                "depth-estimation",
                model=model_name,
                device=0 if device == "cuda" else -1
            )
        
        print("Model loaded successfully!")
    
    def _get_local_model_path(self, model_name):
        """Get local path for cached model."""
        if not _models_cache.exists():
            return None
        
        # HuggingFace cache format: models--{org}--{model}
        cache_name = model_name.replace("/", "--")
        model_path = _models_cache / "models--" / cache_name
        
        return model_path if model_path.exists() else None
    
    def estimate_depth_from_image(self, image_path):
        """
        Estimate depth from a single image file.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            tuple: (original_image, depth_map, normalized_depth)
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_image = np.array(image)
        
        # Estimate depth
        depth_result = self.depth_estimator(image)
        depth_map = np.array(depth_result["depth"])
        
        # Normalize depth for visualization
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return original_image, depth_map, depth_normalized
    
    def estimate_depth_from_array(self, image_array):
        """
        Estimate depth from numpy array (useful for video frames).
        
        Args:
            image_array (np.ndarray): Input image as numpy array
            
        Returns:
            tuple: (depth_map, normalized_depth)
        """
        # Convert numpy array to PIL Image
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        image = Image.fromarray(image_array)
        
        # Estimate depth
        depth_result = self.depth_estimator(image)
        depth_map = np.array(depth_result["depth"])
        
        # Normalize depth for visualization
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return depth_map, depth_normalized
    
    def process_video_stream(self, source=0, output_path=None):
        """
        Process video stream (webcam or video file) for real-time depth estimation.
        
        Args:
            source: Video source (0 for webcam, or path to video file)
            output_path (str): Optional path to save output video
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        print("Starting video processing. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Estimate depth
                _, depth_normalized = self.estimate_depth_from_array(rgb_frame)
                
                # Convert depth to color map
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
                
                # Combine original and depth images
                combined = np.hstack([frame, depth_colored])
                
                # Display result
                cv2.imshow('Original | Depth', combined)
                
                # Save frame if writer is available
                if writer:
                    writer.write(combined)
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
    
    def visualize_depth(self, original_image, depth_map, save_path=None):
        """
        Create a visualization of original image and depth map side by side.
        
        Args:
            original_image (np.ndarray): Original RGB image
            depth_map (np.ndarray): Depth map
            save_path (str): Optional path to save visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Depth map
        depth_vis = axes[1].imshow(depth_map, cmap='plasma')
        axes[1].set_title('Depth Map')
        axes[1].axis('off')
        
        # Add colorbar
        plt.colorbar(depth_vis, ax=axes[1], shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        return fig


def main():
    """Example usage of the DepthEstimator class."""
    
    # Initialize depth estimator
    estimator = DepthEstimator()
    
    # Example: Process a single image (you would need to provide an actual image path)
    # original, depth, depth_norm = estimator.estimate_depth_from_image("path/to/image.jpg")
    # estimator.visualize_depth(original, depth, "depth_output.png")
    
    # Example: Process webcam stream
    # estimator.process_video_stream(source=0)
    
    print("DepthEstimator initialized successfully!")
    print("Use the Streamlit app for interactive demo.")


if __name__ == "__main__":
    main()