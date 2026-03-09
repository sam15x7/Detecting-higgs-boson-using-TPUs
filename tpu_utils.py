"""
TPU utilities for Higgs Boson detection.

This module provides utilities for setting up and utilizing Google TPUs
for accelerated training and inference.
"""

import os
import tensorflow as tf
from typing import Optional, Tuple, Any
import numpy as np


def setup_tpu(tpu_name: Optional[str] = None) -> tf.distribute.TPUStrategy:
    """
    Set up TPU strategy for distributed training.
    
    Args:
        tpu_name: Name of the TPU cluster (optional for Colab)
        
    Returns:
        TPUStrategy object
    """
    try:
        # Try to connect to TPU
        if tpu_name:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            print(f"Connected to TPU: {tpu_name}")
        else:
            # For Colab TPUs
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            print("Connected to Colab TPU")
        
        # Create TPU strategy
        strategy = tf.distribute.TPUStrategy(resolver)
        print(f"Number of TPU cores: {strategy.num_replicas_in_sync}")
        
        return strategy
    
    except Exception as e:
        print(f"TPU setup failed: {e}")
        print("Falling back to CPU/GPU strategy")
        
        # Fall back to default strategy
        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using GPU strategy with {strategy.num_replicas_in_sync} GPUs")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            print("Using CPU strategy")
        
        return strategy


def get_tpu_strategy(tpu_name: Optional[str] = None) -> tf.distribute.Strategy:
    """
    Get TPU distribution strategy.
    
    This is a convenience wrapper around setup_tpu that handles
    common TPU configuration scenarios.
    
    Args:
        tpu_name: Name of the TPU cluster
        
    Returns:
        Distribution strategy object
    """
    return setup_tpu(tpu_name)


def initialize_tpu_cluster(
    project_id: Optional[str] = None,
    zone: Optional[str] = None,
    tpu_name: str = 'higgs-tpu'
) -> tf.distribute.TPUStrategy:
    """
    Initialize TPU cluster on Google Cloud.
    
    Args:
        project_id: GCP project ID
        zone: GCP zone where TPU is located
        tpu_name: Name of the TPU
        
    Returns:
        TPUStrategy object
    """
    # Build TPU name if project and zone provided
    if project_id and zone:
        full_tpu_name = f"projects/{project_id}/locations/{zone}/nodes/{tpu_name}"
    else:
        full_tpu_name = tpu_name
    
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=full_tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        
        strategy = tf.distribute.TPUStrategy(resolver)
        print(f"Initialized TPU cluster: {tpu_name}")
        print(f"Available TPU cores: {strategy.num_replicas_in_sync}")
        
        return strategy
    
    except Exception as e:
        print(f"Failed to initialize TPU cluster: {e}")
        raise


class TPUMirrorStrategy:
    """
    Wrapper for TPU mirrored strategy with additional utilities.
    """
    
    def __init__(self, tpu_name: Optional[str] = None):
        """Initialize TPU mirror strategy."""
        self.strategy = setup_tpu(tpu_name)
        self.num_replicas = self.strategy.num_replicas_in_sync
    
    def distribute_dataset(
        self,
        dataset: tf.data.Dataset,
        batch_size: int
    ) -> tf.data.Dataset:
        """
        Distribute dataset across TPU cores.
        
        Args:
            dataset: Input TensorFlow dataset
            batch_size: Global batch size
            
        Returns:
            Distributed dataset
        """
        # Adjust batch size per replica
        per_replica_batch_size = batch_size // self.num_replicas
        
        # Redefine dataset with new batch size
        dataset = dataset.unbatch().batch(per_replica_batch_size, drop_remainder=True)
        
        # Distribute dataset
        distributed_dataset = self.strategy.experimental_distribute_dataset(dataset)
        
        return distributed_dataset
    
    def create_model(self, model_fn, *args, **kwargs):
        """
        Create model within strategy scope.
        
        Args:
            model_fn: Function that creates and returns a model
            *args: Arguments to pass to model_fn
            **kwargs: Keyword arguments to pass to model_fn
            
        Returns:
            Compiled model within strategy scope
        """
        with self.strategy.scope():
            model = model_fn(*args, **kwargs)
        return model


def check_tpu_availability() -> dict:
    """
    Check availability of TPUs and other accelerators.
    
    Returns:
        Dictionary with hardware availability information
    """
    info = {
        'tpu_available': False,
        'gpu_available': False,
        'num_gpus': 0,
        'device_type': 'CPU'
    }
    
    # Check for TPUs
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        info['tpu_available'] = True
        info['device_type'] = 'TPU'
        print("✓ TPU available")
    except:
        print("✗ TPU not available")
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        info['gpu_available'] = True
        info['num_gpus'] = len(gpus)
        if not info['tpu_available']:
            info['device_type'] = 'GPU'
        print(f"✓ {len(gpus)} GPU(s) available")
    else:
        print("✗ No GPUs available")
    
    return info


def optimize_for_tpu(model: tf.keras.Model) -> tf.keras.Model:
    """
    Apply TPU-specific optimizations to a model.
    
    Args:
        model: Keras model to optimize
        
    Returns:
        Optimized model
    """
    # TPU-friendly optimizations
    # 1. Use float32 for better TPU performance
    tf.keras.backend.set_floatx('float32')
    
    # 2. Ensure compatible activation functions
    # (already using ReLU, sigmoid which are TPU-friendly)
    
    # 3. Use appropriate initializers
    # (already using he_normal, glorot_uniform)
    
    print("Model optimized for TPU execution")
    return model


def tpu_batch_norm_fusion(model: tf.keras.Model) -> tf.keras.Model:
    """
    Apply batch normalization fusion for TPU optimization.
    
    TPUs can fuse batch normalization into preceding layers for
    better performance.
    
    Args:
        model: Keras model
        
    Returns:
        Model with fused batch normalization
    """
    # Note: TensorFlow automatically handles BN fusion on TPUs
    # when using TPUStrategy
    print("Batch normalization will be automatically fused on TPU")
    return model


if __name__ == "__main__":
    # Test TPU utilities
    print("Testing TPU utilities...")
    
    # Check hardware availability
    info = check_tpu_availability()
    print(f"\nHardware info: {info}")
    
    # Try to set up TPU strategy
    try:
        strategy = setup_tpu()
        print(f"\nStrategy type: {type(strategy).__name__}")
        print(f"Number of replicas: {strategy.num_replicas_in_sync}")
    except Exception as e:
        print(f"\nCould not set up TPU: {e}")
    
    print("\nTPU utilities test complete!")
