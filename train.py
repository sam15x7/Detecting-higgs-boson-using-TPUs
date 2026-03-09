"""
Training pipeline for Higgs Boson detection on TPUs.

This module provides the main training loop and utilities for training
models on TPUs with support for distributed training, mixed precision,
and experiment tracking.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from .tpu_utils import setup_tpu, get_tpu_strategy
from .evaluate import AMSMetric


class Trainer:
    """
    Trainer class for Higgs Boson detection models.
    
    Provides a unified interface for training models on TPUs with
    support for distributed training, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss_fn: str = 'binary_crossentropy',
        metrics: list = None,
        tpu_name: Optional[str] = None,
        output_dir: str = 'models/',
        run_name: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Keras model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            optimizer: Optimizer name or instance
            learning_rate: Learning rate
            loss_fn: Loss function name or instance
            metrics: List of metrics to track
            tpu_name: Name of TPU cluster (optional)
            output_dir: Directory for saving checkpoints
            run_name: Name for this training run
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup TPU strategy
        self.strategy = get_tpu_strategy(tpu_name)
        self.num_replicas = self.strategy.num_replicas_in_sync
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"higgs_run_{timestamp}"
        self.run_name = run_name
        
        # Setup optimizer
        if isinstance(optimizer, str):
            if optimizer.lower() == 'adam':
                self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer.lower() == 'sgd':
                self.optimizer = keras.optimizers.SGD(
                    learning_rate=learning_rate, momentum=0.9
                )
            elif optimizer.lower() == 'rmsprop':
                self.optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            self.optimizer = optimizer
        
        # Setup loss function
        if isinstance(loss_fn, str):
            self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
        else:
            self.loss_fn = loss_fn
        
        # Setup metrics
        self.metrics = metrics or ['accuracy']
        if 'ams' not in [m.lower() if isinstance(m, str) else type(m).__name__.lower() 
                         for m in self.metrics]:
            self.metrics.append(AMSMetric())
        
        # Training history
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'ams': [],
            'val_ams': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_ams = -float('inf')
    
    def compile_model(self):
        """Compile the model within strategy scope."""
        with self.strategy.scope():
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                metrics=self.metrics
            )
        print("Model compiled successfully")
    
    def create_callbacks(
        self,
        epochs: int,
        save_best_only: bool = True,
        early_stopping: bool = True,
        reduce_lr: bool = True,
        patience: int = 5
    ) -> List[callbacks.Callback]:
        """
        Create training callbacks.
        
        Args:
            epochs: Total number of epochs
            save_best_only: Whether to save only best model
            early_stopping: Whether to use early stopping
            reduce_lr: Whether to reduce LR on plateau
            patience: Patience for early stopping
            
        Returns:
            List of callbacks
        """
        callback_list = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(
            self.output_dir,
            f"{self.run_name}_best_model.h5"
        )
        callback_list.append(
            callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=save_best_only,
                save_weights_only=False,
                verbose=1
            )
        )
        
        # Early stopping
        if early_stopping:
            callback_list.append(
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                )
            )
        
        # Reduce learning rate on plateau
        if reduce_lr:
            callback_list.append(
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=patience // 2,
                    min_lr=1e-7,
                    verbose=1
                )
            )
        
        # TensorBoard
        log_dir = os.path.join(self.output_dir, 'logs', self.run_name)
        callback_list.append(
            callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True
            )
        )
        
        return callback_list
    
    def train(
        self,
        epochs: int = 10,
        batch_size: int = 1024,
        callbacks_list: Optional[List[callbacks.Callback]] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks_list: Optional list of callbacks
            verbose: Verbosity mode (0, 1, or 2)
            
        Returns:
            Training history
        """
        print(f"\nStarting training on {self.num_replicas} TPU replicas")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print(f"Run name: {self.run_name}")
        print("-" * 60)
        
        # Compile model if not already compiled
        if not self.model.optimizer:
            self.compile_model()
        
        # Create default callbacks if none provided
        if callbacks_list is None:
            callbacks_list = self.create_callbacks(epochs)
        
        # Calculate steps per epoch
        train_steps = len(list(self.train_dataset))
        val_steps = len(list(self.val_dataset))
        
        # Train the model
        start_time = time.time()
        
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Total training time: {training_time:.2f} seconds")
        print(f"Average time per epoch: {training_time / epochs:.2f} seconds")
        print("=" * 60)
        
        # Store history
        self.history = history.history
        
        return history
    
    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating model on test set...")
        
        test_steps = len(list(test_dataset))
        results = self.model.evaluate(test_dataset, steps=test_steps, verbose=1)
        
        # Create results dictionary
        metric_names = ['loss'] + [m.name if hasattr(m, 'name') else m 
                                   for m in self.model.metrics]
        evaluation = dict(zip(metric_names, results))
        
        print("\nTest Results:")
        for key, value in evaluation.items():
            print(f"  {key}: {value:.4f}")
        
        return evaluation
    
    def save_model(self, filename: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            filename: Optional filename (default: run_name + timestamp)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.run_name}_final_{timestamp}.h5"
        
        filepath = os.path.join(self.output_dir, filename)
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to saved model
        """
        with self.strategy.scope():
            self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")


def train_on_tpu(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    tpu_name: Optional[str] = None,
    output_dir: str = 'models/',
    **kwargs
) -> Trainer:
    """
    Convenience function to train a model on TPU.
    
    Args:
        model: Keras model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        tpu_name: Name of TPU cluster
        output_dir: Directory for saving checkpoints
        **kwargs: Additional arguments for Trainer
        
    Returns:
        Trained Trainer object
    """
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=learning_rate,
        tpu_name=tpu_name,
        output_dir=output_dir,
        **kwargs
    )
    
    # Train
    trainer.train(epochs=epochs, batch_size=batch_size)
    
    return trainer


if __name__ == "__main__":
    # Example usage
    print("Testing Trainer class...")
    
    # Create dummy model
    from .model import HiggsClassifier
    model = HiggsClassifier(input_dim=28, hidden_dims=[128, 64]).build_model()
    
    # Create dummy datasets
    import numpy as np
    X_train = np.random.randn(1000, 28).astype(np.float32)
    y_train = np.random.randint(0, 2, 1000).astype(np.float32)
    X_val = np.random.randn(200, 28).astype(np.float32)
    y_val = np.random.randint(0, 2, 200).astype(np.float32)
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
    
    # Create trainer (will use CPU/GPU since no TPU available in test)
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        learning_rate=0.001,
        output_dir='test_models/'
    )
    
    print("Trainer initialized successfully")
    print(f"Number of replicas: {trainer.num_replicas}")
