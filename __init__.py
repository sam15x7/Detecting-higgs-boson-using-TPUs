"""
Higgs Boson Detection with TPUs

A comprehensive machine learning project for detecting Higgs boson particles
using Google Tensor Processing Units (TPUs).
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_loader import HiggsDataset, DataTransforms
from .model import HiggsClassifier, ResNetHiggs, DenseNetHiggs
from .tpu_utils import get_tpu_strategy, setup_tpu, initialize_tpu_cluster
from .train import Trainer, train_on_tpu
from .evaluate import AMSMetric, evaluate_model

__all__ = [
    "HiggsDataset",
    "DataTransforms",
    "HiggsClassifier",
    "ResNetHiggs",
    "DenseNetHiggs",
    "get_tpu_strategy",
    "setup_tpu",
    "initialize_tpu_cluster",
    "Trainer",
    "train_on_tpu",
    "AMSMetric",
    "evaluate_model",
]
