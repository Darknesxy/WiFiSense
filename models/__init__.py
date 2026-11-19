"""
WiFiSense Models Package
Implements Deep Learning architectures (1D-CNN, 2D-CNN, Attention-CNN) and ML baseline comparisons.
"""

from .cnn_models import CSI_1D_CNN, CSI_2D_CNN, CSI_CNN_Attention
from .training import train_cnn_model, CSIDataset, prepare_dataloaders
from .evaluation import evaluate_model, compare_all_models, benchmark_model_inference_latency

__all__ = [
    "CSI_1D_CNN",
    "CSI_2D_CNN",
    "CSI_CNN_Attention",
    "train_cnn_model",
    "CSIDataset",
    "prepare_dataloaders",
    "evaluate_model",
    "compare_all_models",
    "benchmark_model_inference_latency"
]
