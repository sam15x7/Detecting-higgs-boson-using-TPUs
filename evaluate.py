"""
Evaluation metrics for Higgs Boson detection.

This module provides physics-specific evaluation metrics including
the Approximate Median Significance (AMS) metric used in the Higgs
Boson Machine Learning Challenge.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional


class AMSMetric(keras.metrics.Metric):
    """
    Approximate Median Significance (AMS) metric.
    
    The AMS is a custom metric designed for the Higgs Boson ML Challenge
    that approximates the significance of a signal discovery in particle
    physics experiments.
    
    Formula:
        AMS = sqrt(2 * ((s + b) * log(1 + s/b) - s))
    
    where:
        s = weighted signal events
        b = weighted background events
    """
    
    def __init__(self, name='ams', br: float = 0.1, **kwargs):
        """
        Initialize AMS metric.
        
        Args:
            name: Metric name
            br: Signal to background ratio (b_r in original formula)
            **kwargs: Additional arguments for Metric
        """
        super(AMSMetric, self).__init__(name=name, **kwargs)
        self.br = br
        self.true_positives = self.add_weight(
            name='true_positives', initializer='zeros'
        )
        self.false_positives = self.add_weight(
            name='false_positives', initializer='zeros'
        )
        self.total_samples = self.add_weight(
            name='total_samples', initializer='zeros'
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update metric state.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted probabilities
            sample_weight: Optional sample weights
        """
        # Convert to numpy if tensor
        if hasattr(y_true, 'numpy'):
            y_true = y_true.numpy()
        if hasattr(y_pred, 'numpy'):
            y_pred = y_pred.numpy()
        
        # Apply threshold to get binary predictions
        y_pred_binary = (y_pred >= 0.5).astype(np.float32)
        
        # Calculate true positives (signal correctly identified)
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        
        # Calculate false positives (background misidentified as signal)
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        
        # Update counts
        self.true_positives.assign_add(float(tp))
        self.false_positives.assign_add(float(fp))
        self.total_samples.assign_add(float(len(y_true)))
    
    def result(self):
        """Calculate and return AMS score."""
        s = float(self.true_positives.read_value())
        b = float(self.false_positives.read_value())
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        
        # Calculate AMS with regularization
        if b < 1e-6:
            return 0.0
        
        # Original AMS formula with regularization
        br = self.br
        s_reg = s
        b_reg = b + 1.0 / (br * br)
        
        ams = np.sqrt(
            2 * (
                (s_reg + b_reg) * np.log(1 + s_reg / b_reg) - s_reg
            )
        )
        
        return tf.convert_to_tensor(ams, dtype=tf.float32)
    
    def reset_state(self):
        """Reset metric state."""
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.total_samples.assign(0.0)
    
    def get_config(self):
        """Get metric configuration."""
        config = super(AMSMetric, self).get_config()
        config.update({'br': self.br})
        return config


def calculate_ams(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    br: float = 0.1
) -> float:
    """
    Calculate Approximate Median Significance.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities
        threshold: Classification threshold
        br: Signal to background ratio
        
    Returns:
        AMS score
    """
    # Apply threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Count signal and background
    signal_mask = y_true == 1
    background_mask = y_true == 0
    
    # True positives (signal selected)
    s = np.sum(y_pred_binary[signal_mask] == 1)
    
    # False positives (background selected)
    b = np.sum(y_pred_binary[background_mask] == 1)
    
    # Regularization term
    br_factor = 1.0 / (br * br)
    
    # Calculate AMS
    if b < 1e-6:
        return 0.0
    
    s_reg = s
    b_reg = b + br_factor
    
    ams = np.sqrt(
        2 * (
            (s_reg + b_reg) * np.log(1 + s_reg / b_reg) - s_reg
        )
    )
    
    return ams


def calculate_ams_optimized(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    br: float = 0.1
) -> Tuple[float, float]:
    """
    Calculate optimized AMS by finding the best threshold.
    
    This function scans through different thresholds to find the one
    that maximizes the AMS score.
    
    Args:
        y_true: True labels (0 or 1)
        y_prob: Predicted probabilities
        br: Signal to background ratio
        
    Returns:
        Tuple of (best_ams, best_threshold)
    """
    # Sort by predicted probability
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]
    
    # Cumulative sums
    n_signal = np.sum(y_true == 1)
    n_background = np.sum(y_true == 0)
    
    cum_signal = np.cumsum(y_true_sorted)
    cum_background = np.cumsum(1 - y_true_sorted)
    
    # Calculate AMS at each threshold
    br_factor = 1.0 / (br * br)
    
    ams_scores = []
    for i in range(len(y_true)):
        s = cum_signal[i]
        b = cum_background[i]
        
        if b < 1e-6:
            ams_scores.append(0.0)
        else:
            s_reg = s
            b_reg = b + br_factor
            
            ams = np.sqrt(
                2 * (
                    (s_reg + b_reg) * np.log(1 + s_reg / b_reg) - s_reg
                )
            )
            ams_scores.append(ams)
    
    # Find best threshold
    best_idx = np.argmax(ams_scores)
    best_ams = ams_scores[best_idx]
    best_threshold = y_prob_sorted[best_idx]
    
    return best_ams, best_threshold


class PrecisionRecallCurve:
    """Utility class for computing precision-recall curves."""
    
    def __init__(self, y_true: np.ndarray, y_prob: np.ndarray):
        """
        Initialize PR curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
        """
        self.y_true = y_true
        self.y_prob = y_prob
        
        # Sort by probability
        self.sorted_indices = np.argsort(y_prob)[::-1]
        self.y_true_sorted = y_true[self.sorted_indices]
        
        # Precompute cumulative sums
        self.n_samples = len(y_true)
        self.cum_tp = np.cumsum(self.y_true_sorted)
        self.cum_fp = np.cumsum(1 - self.y_true_sorted)
    
    def get_precision_recall(
        self,
        threshold: float
    ) -> Tuple[float, float]:
        """
        Get precision and recall at a given threshold.
        
        Args:
            threshold: Classification threshold
            
        Returns:
            Tuple of (precision, recall)
        """
        # Get predictions above threshold
        mask = self.y_prob >= threshold
        n_pred = np.sum(mask)
        
        if n_pred == 0:
            return 1.0, 0.0
        
        tp = np.sum((self.y_true == 1) & mask)
        fp = np.sum((self.y_true == 0) & mask)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / np.sum(self.y_true == 1) if np.sum(self.y_true == 1) > 0 else 0.0
        
        return precision, recall
    
    def get_curve(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get full precision-recall curve.
        
        Args:
            n_points: Number of points on the curve
            
        Returns:
            Tuple of (thresholds, precisions, recalls)
        """
        thresholds = np.linspace(0, 1, n_points)
        precisions = []
        recalls = []
        
        for thresh in thresholds:
            p, r = self.get_precision_recall(thresh)
            precisions.append(p)
            recalls.append(r)
        
        return thresholds, np.array(precisions), np.array(recalls)
    
    def get_auc(self) -> float:
        """
        Calculate area under the precision-recall curve.
        
        Returns:
            AUC-PR score
        """
        thresholds, precisions, recalls = self.get_curve()
        
        # Use trapezoidal rule
        auc = np.trapz(precisions, recalls)
        
        return auc


def evaluate_model(
    model: keras.Model,
    test_dataset: tf.data.Dataset,
    include_ams: bool = True
) -> dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained Keras model
        test_dataset: Test dataset
        include_ams: Whether to calculate AMS metric
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating model...")
    
    # Get predictions
    y_true_list = []
    y_pred_list = []
    
    for X_batch, y_batch in test_dataset:
        y_pred = model.predict(X_batch, verbose=0)
        y_true_list.append(y_batch.numpy())
        y_pred_list.append(y_pred.flatten())
    
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    
    # Calculate basic metrics
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred_binary)
    
    # True positives, false positives, etc.
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    # Precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    # Calculate AMS
    if include_ams:
        ams_standard = calculate_ams(y_true, y_pred, threshold=0.5)
        ams_optimized, optimal_threshold = calculate_ams_optimized(y_true, y_pred)
        
        results['ams_standard'] = ams_standard
        results['ams_optimized'] = ams_optimized
        results['optimal_threshold'] = optimal_threshold
        
        print(f"\nAMS Metrics:")
        print(f"  Standard AMS (threshold=0.5): {ams_standard:.4f}")
        print(f"  Optimized AMS: {ams_optimized:.4f} (threshold={optimal_threshold:.4f})")
    
    print(f"\nEvaluation Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return results


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.rand(n_samples)
    
    # Test AMS calculation
    ams = calculate_ams(y_true, y_prob)
    print(f"Standard AMS: {ams:.4f}")
    
    ams_opt, thresh = calculate_ams_optimized(y_true, y_prob)
    print(f"Optimized AMS: {ams_opt:.4f} (threshold={thresh:.4f})")
    
    # Test PR curve
    pr_curve = PrecisionRecallCurve(y_true, y_prob)
    auc_pr = pr_curve.get_auc()
    print(f"AUC-PR: {auc_pr:.4f}")
    
    # Test AMS metric
    ams_metric = AMSMetric()
    ams_metric.update_state(y_true, y_prob)
    print(f"AMS Metric result: {ams_metric.result().numpy():.4f}")
    
    print("\nAll metrics tested successfully!")
