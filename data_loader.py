"""
Data loading and preprocessing for Higgs Boson detection.

This module provides efficient data loading pipelines optimized for large-scale
particle physics datasets, with support for TPU acceleration.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split


class DataTransforms:
    """Data transformation utilities for Higgs dataset preprocessing."""
    
    def __init__(self, feature_columns: list = None):
        """
        Initialize data transforms.
        
        Args:
            feature_columns: List of feature column names to use
        """
        self.feature_columns = feature_columns
        self.feature_means = None
        self.feature_stds = None
        
    def fit(self, features: np.ndarray) -> 'DataTransforms':
        """
        Fit the transforms on training data.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            self
        """
        self.feature_means = np.mean(features, axis=0)
        self.feature_stds = np.std(features, axis=0) + 1e-8
        return self
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted parameters.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            Transformed features
        """
        if self.feature_means is None or self.feature_stds is None:
            raise ValueError("Transform must be fitted before use")
        return (features - self.feature_means) / self.feature_stds
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(features).transform(features)


class HiggsDataset:
    """
    Higgs Boson dataset loader and processor.
    
    This class handles loading the HIGGS dataset from UCI Machine Learning
    Repository, performing preprocessing, and creating TensorFlow datasets
    optimized for TPU training.
    """
    
    FEATURE_COLUMNS = [
        'lepton_pT', 'lepton_eta', 'lepton_phi',
        'missing_energy_magnitude', 'missing_energy_phi',
        'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_btag',
        'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_btag',
        'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_btag',
        'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_btag',
        'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
    ]
    
    def __init__(
        self,
        data_path: str = 'data/',
        feature_columns: list = None,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the Higgs dataset.
        
        Args:
            data_path: Path to the data directory
            feature_columns: List of feature columns to use (default: all 28)
            test_size: Fraction of data to use for testing
            val_size: Fraction of training data to use for validation
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.feature_columns = feature_columns or self.FEATURE_COLUMNS
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.transforms = DataTransforms(feature_columns)
        
    def load_csv(
        self,
        filename: str,
        nrows: Optional[int] = None,
        chunksize: int = 100000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from CSV file in chunks to handle large files.
        
        Args:
            filename: Name of the CSV file
            nrows: Maximum number of rows to load (None for all)
            chunksize: Number of rows per chunk
            
        Returns:
            Tuple of (features, labels) arrays
        """
        filepath = os.path.join(self.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Read data in chunks to handle large files
        chunks = []
        total_rows = 0
        
        for chunk in pd.read_csv(filepath, header=None, chunksize=chunksize):
            if nrows and total_rows >= nrows:
                break
                
            chunks.append(chunk)
            total_rows += len(chunk)
            
            if nrows and total_rows >= nrows:
                chunks[-1] = chunks[-1].iloc[:nrows - (total_rows - len(chunk))]
                break
        
        df = pd.concat(chunks, ignore_index=True)
        
        # Extract features and labels
        labels = df.iloc[:, 0].values.astype(np.float32)
        features = df.iloc[:, 1:].values.astype(np.float32)
        
        return features, labels
    
    def prepare_data(
        self,
        sample_size: Optional[int] = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare train/validation/test splits.
        
        Args:
            sample_size: Optional subsample size for quick experiments
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        print("Loading training data...")
        features, labels = self.load_csv('HIGGS.csv', nrows=sample_size)
        
        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        # Split train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state,
            stratify=y_temp
        )
        
        # Fit transforms on training data
        print("Fitting data transforms...")
        self.transforms.fit(X_train)
        
        # Apply transforms
        X_train = self.transforms.transform(X_train)
        X_val = self.transforms.transform(X_val)
        X_test = self.transforms.transform(X_test)
        
        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)
        
        print(f"Data preparation complete:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Test samples: {len(X_test)}")
        
        return {
            'train': self.train_data,
            'val': self.val_data,
            'test': self.test_data
        }
    
    def create_tf_dataset(
        self,
        split: str = 'train',
        batch_size: int = 1024,
        shuffle: bool = True,
        prefetch: bool = True,
        drop_remainder: bool = True
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset for the specified split.
        
        Args:
            split: Data split ('train', 'val', or 'test')
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            prefetch: Whether to prefetch batches
            drop_remainder: Whether to drop incomplete batches
            
        Returns:
            TensorFlow Dataset object
        """
        if split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        elif split == 'test':
            data = self.test_data
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        features, labels = data
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        
        if shuffle and split == 'train':
            dataset = dataset.shuffle(buffer_size=len(features))
        
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_loaders(
        self,
        batch_size: int = 1024,
        sample_size: Optional[int] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Get train, validation, and test data loaders.
        
        Args:
            batch_size: Batch size for training
            sample_size: Optional subsample size
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        self.prepare_data(sample_size=sample_size)
        
        train_loader = self.create_tf_dataset(
            'train', batch_size=batch_size, shuffle=True
        )
        val_loader = self.create_tf_dataset(
            'val', batch_size=batch_size, shuffle=False
        )
        test_loader = self.create_tf_dataset(
            'test', batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    @property
    def input_dim(self) -> int:
        """Return the number of input features."""
        return len(self.feature_columns)


def create_sample_data(
    n_samples: int = 10000,
    n_features: int = 28,
    signal_fraction: float = 0.5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic sample data for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        signal_fraction: Fraction of signal events
        seed: Random seed
        
    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(seed)
    
    # Generate synthetic features
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate labels with some correlation to features
    signal_prob = 1 / (1 + np.exp(-np.sum(features[:, :5], axis=1)))
    labels = (np.random.rand(n_samples) < signal_prob).astype(np.float32)
    
    return features, labels


if __name__ == "__main__":
    # Example usage
    print("Testing HiggsDataset...")
    
    # Create sample data for testing
    features, labels = create_sample_data(n_samples=1000)
    print(f"Generated {len(features)} samples with {features.shape[1]} features")
    
    # Test transforms
    transforms = DataTransforms()
    transformed = transforms.fit_transform(features)
    print(f"Transformed data shape: {transformed.shape}")
    print(f"Mean after transform: {np.mean(transformed, axis=0)[:5]}")
    print(f"Std after transform: {np.std(transformed, axis=0)[:5]}")
