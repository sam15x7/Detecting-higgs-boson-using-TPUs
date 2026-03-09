"""
Unit tests for Higgs Boson detection models.
"""

import pytest
import numpy as np
import tensorflow as tf
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import HiggsClassifier, ResNetHiggs, DenseNetHiggs, create_model
from src.data_loader import DataTransforms, create_sample_data
from src.evaluate import AMSMetric, calculate_ams


class TestModels:
    """Test model architectures."""
    
    def test_higgs_classifier_creation(self):
        """Test MLP model creation."""
        model = HiggsClassifier(input_dim=28, hidden_dims=[128, 64])
        keras_model = model.build_model()
        
        assert keras_model is not None
        assert keras_model.input_shape == (None, 28)
        assert keras_model.output_shape == (None, 1)
    
    def test_resnet_higgs_creation(self):
        """Test ResNet model creation."""
        model = ResNetHiggs(input_dim=28, num_blocks=2, units_per_block=64)
        keras_model = model.build_model()
        
        assert keras_model is not None
        assert keras_model.input_shape == (None, 28)
        assert keras_model.output_shape == (None, 1)
    
    def test_densenet_higgs_creation(self):
        """Test DenseNet model creation."""
        model = DenseNetHiggs(input_dim=28, growth_rate=16, num_dense_blocks=2)
        keras_model = model.build_model()
        
        assert keras_model is not None
        assert keras_model.input_shape == (None, 28)
        assert keras_model.output_shape == (None, 1)
    
    def test_model_forward_pass(self):
        """Test forward pass through models."""
        model = HiggsClassifier(input_dim=28, hidden_dims=[64, 32]).build_model()
        
        # Create dummy input
        dummy_input = np.random.randn(32, 28).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        
        assert output.shape == (32, 1)
        assert np.all((output >= 0) & (output <= 1))
    
    def test_create_model_factory(self):
        """Test model factory function."""
        for model_type in ['mlp', 'resnet', 'densenet']:
            model = create_model(model_type=model_type, input_dim=28)
            assert model is not None
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError):
            create_model(model_type='invalid', input_dim=28)


class TestDataLoader:
    """Test data loading utilities."""
    
    def test_data_transforms(self):
        """Test data transformation."""
        transforms = DataTransforms()
        
        # Create sample data
        data = np.random.randn(100, 10).astype(np.float32)
        
        # Fit and transform
        transformed = transforms.fit_transform(data)
        
        assert transformed.shape == data.shape
        assert np.allclose(np.mean(transformed, axis=0), 0, atol=1e-6)
        assert np.allclose(np.std(transformed, axis=0), 1, atol=1e-6)
    
    def test_create_sample_data(self):
        """Test sample data generation."""
        features, labels = create_sample_data(n_samples=100, n_features=28)
        
        assert features.shape == (100, 28)
        assert labels.shape == (100,)
        assert features.dtype == np.float32
        assert labels.dtype == np.float32


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_ams_metric_creation(self):
        """Test AMS metric creation."""
        metric = AMSMetric()
        assert metric is not None
        assert metric.name == 'ams'
    
    def test_ams_metric_update(self):
        """Test AMS metric state update."""
        metric = AMSMetric()
        
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([0.9, 0.2, 0.8, 0.3, 0.7])
        
        metric.update_state(y_true, y_pred)
        result = metric.result().numpy()
        
        assert result >= 0
    
    def test_calculate_ams(self):
        """Test AMS calculation."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0])
        y_pred = np.array([0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.4, 0.6])
        
        ams = calculate_ams(y_true, y_pred)
        assert ams >= 0
    
    def test_calculate_ams_perfect(self):
        """Test AMS with perfect predictions."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        
        ams = calculate_ams(y_true, y_pred)
        # Perfect classification should give high AMS
        assert ams > 0


class TestIntegration:
    """Integration tests."""
    
    def test_model_training_step(self):
        """Test a single training step."""
        # Create model
        model = HiggsClassifier(input_dim=28, hidden_dims=[32, 16]).build_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Create sample data
        X, y = create_sample_data(n_samples=100, n_features=28)
        
        # Train for one epoch
        history = model.fit(X, y, epochs=1, verbose=0)
        
        assert 'loss' in history.history
        assert 'accuracy' in history.history
    
    def test_full_pipeline(self):
        """Test complete training pipeline."""
        # Create model
        model = HiggsClassifier(input_dim=28, hidden_dims=[64, 32]).build_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AMSMetric()])
        
        # Create sample data
        X_train, y_train = create_sample_data(n_samples=500, n_features=28)
        X_val, y_val = create_sample_data(n_samples=100, n_features=28)
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=2,
            verbose=0
        )
        
        # Check training occurred
        assert len(history.history['loss']) == 2
        assert len(history.history['val_loss']) == 2
        
        # Evaluate
        results = model.evaluate(X_val, y_val, verbose=0)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
