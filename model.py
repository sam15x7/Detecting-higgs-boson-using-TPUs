"""
Neural network architectures for Higgs Boson detection.

This module provides various deep learning model architectures optimized
for TPU acceleration, including MLPs, ResNet-style, and DenseNet-style models.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import List, Optional, Dict, Any


class HiggsClassifier(Model):
    """
    Multi-layer perceptron classifier for Higgs Boson detection.
    
    A flexible feedforward neural network with configurable hidden layers,
    dropout, and batch normalization.
    """
    
    def __init__(
        self,
        input_dim: int = 28,
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        activation: str = 'relu',
        output_activation: str = 'sigmoid'
    ):
        """
        Initialize the Higgs classifier.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
        """
        super(HiggsClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Input layer
        self.input_layer = layers.Input(shape=(input_dim,))
        
        # Hidden layers
        self.dense_layers = []
        self.bn_layers = []
        self.dropout_layers = []
        
        for dim in hidden_dims:
            self.dense_layers.append(
                layers.Dense(dim, activation=activation, kernel_initializer='he_normal')
            )
            if use_batch_norm:
                self.bn_layers.append(layers.BatchNormalization())
            self.dropout_layers.append(layers.Dropout(dropout_rate))
        
        # Output layer
        self.output_layer = layers.Dense(
            1, activation=output_activation,
            kernel_initializer='glorot_uniform'
        )
    
    def call(self, inputs, training=False):
        """Forward pass."""
        x = inputs
        
        for i, dense in enumerate(self.dense_layers):
            x = dense(x)
            if self.use_batch_norm and self.bn_layers:
                x = self.bn_layers[i](x, training=training)
            x = self.dropout_layers[i](x, training=training)
        
        return self.output_layer(x)
    
    def build_model(self) -> Model:
        """Build and return the Keras model."""
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs
        
        for i, dense in enumerate(self.dense_layers):
            x = dense(x)
            if self.use_batch_norm and self.bn_layers:
                x = self.bn_layers[i](x)
            x = self.dropout_layers[i](x)
        
        outputs = self.output_layer(x)
        
        return Model(inputs=inputs, outputs=outputs)


class ResidualBlock(layers.Layer):
    """Residual block for ResNet-style architecture."""
    
    def __init__(
        self,
        units: int,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        super(ResidualBlock, self).__init__()
        
        self.dense1 = layers.Dense(
            units, activation='relu', kernel_initializer='he_normal'
        )
        self.dense2 = layers.Dense(
            units, activation=None, kernel_initializer='he_normal'
        )
        self.dropout = layers.Dropout(dropout_rate)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.use_batch_norm = use_batch_norm
        
        # Projection shortcut if dimensions change
        self.project = None
    
    def build(self, input_shape):
        if input_shape[-1] != self.dense2.units:
            self.project = layers.Dense(
                self.dense2.units, activation=None,
                kernel_initializer='he_normal'
            )
    
    def call(self, inputs, training=False):
        identity = inputs
        
        x = self.dense1(inputs)
        if self.use_batch_norm:
            x = self.bn1(x, training=training)
        x = self.dropout(x, training=training)
        
        x = self.dense2(x)
        if self.use_batch_norm:
            x = self.bn2(x, training=training)
        
        # Apply projection if needed
        if self.project is not None:
            identity = self.project(identity)
        
        # Residual connection
        x = layers.add([x, identity])
        x = layers.ReLU()(x)
        
        return x


class ResNetHiggs(Model):
    """
    ResNet-style architecture for Higgs Boson detection.
    
    Implements residual connections to enable deeper networks without
    vanishing gradient problems.
    """
    
    def __init__(
        self,
        input_dim: int = 28,
        initial_units: int = 128,
        num_blocks: int = 3,
        units_per_block: int = 128,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Initialize ResNet Higgs model.
        
        Args:
            input_dim: Number of input features
            initial_units: Units in initial dense layer
            num_blocks: Number of residual blocks
            units_per_block: Units in each residual block
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(ResNetHiggs, self).__init__()
        
        self.initial_dense = layers.Dense(
            initial_units, activation='relu', kernel_initializer='he_normal'
        )
        self.initial_bn = layers.BatchNormalization()
        self.initial_dropout = layers.Dropout(dropout_rate)
        
        self.residual_blocks = []
        for _ in range(num_blocks):
            self.residual_blocks.append(
                ResidualBlock(units_per_block, dropout_rate, use_batch_norm)
            )
        
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.output_layer = layers.Dense(
            1, activation='sigmoid', kernel_initializer='glorot_uniform'
        )
    
    def call(self, inputs, training=False):
        """Forward pass."""
        # Ensure 2D input
        if len(inputs.shape) == 2:
            x = tf.expand_dims(inputs, axis=-1)
        else:
            x = inputs
        
        x = self.initial_dense(x)
        x = self.initial_bn(x, training=training)
        x = self.initial_dropout(x, training=training)
        
        for block in self.residual_blocks:
            x = block(x, training=training)
        
        x = self.global_avg_pool(x)
        return self.output_layer(x)
    
    def build_model(self) -> Model:
        """Build and return the Keras model."""
        inputs = layers.Input(shape=(self.input_dim,))
        
        if len(inputs.shape) == 2:
            x = tf.expand_dims(inputs, axis=-1)
        else:
            x = inputs
        
        x = self.initial_dense(x)
        x = self.initial_bn(x)
        x = self.initial_dropout(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.global_avg_pool(x)
        outputs = self.output_layer(x)
        
        return Model(inputs=inputs, outputs=outputs)


class DenseBlock(layers.Layer):
    """Dense block for DenseNet-style architecture."""
    
    def __init__(
        self,
        growth_rate: int = 32,
        num_layers: int = 4,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        super(DenseBlock, self).__init__()
        
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.layers_list = []
        self.bn_layers = []
        self.dropout_layers = []
        
        for _ in range(num_layers):
            self.layers_list.append(
                layers.Dense(growth_rate, activation='relu', kernel_initializer='he_normal')
            )
            if use_batch_norm:
                self.bn_layers.append(layers.BatchNormalization())
            self.dropout_layers.append(layers.Dropout(dropout_rate))
        
        self.use_batch_norm = use_batch_norm
    
    def call(self, inputs, training=False):
        """Forward pass with concatenation."""
        x = inputs
        
        for i in range(self.num_layers):
            layer_output = self.layers_list[i](x)
            if self.use_batch_norm and self.bn_layers:
                layer_output = self.bn_layers[i](layer_output, training=training)
            layer_output = self.dropout_layers[i](layer_output, training=training)
            
            # Concatenate along feature dimension
            x = layers.concatenate([x, layer_output], axis=-1)
        
        return x


class DenseNetHiggs(Model):
    """
    DenseNet-style architecture for Higgs Boson detection.
    
    Implements dense connections where each layer receives feature maps
    from all preceding layers.
    """
    
    def __init__(
        self,
        input_dim: int = 28,
        initial_units: int = 64,
        growth_rate: int = 32,
        num_dense_blocks: int = 3,
        layers_per_block: int = 4,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Initialize DenseNet Higgs model.
        
        Args:
            input_dim: Number of input features
            initial_units: Units in initial dense layer
            growth_rate: Number of filters added per layer
            num_dense_blocks: Number of dense blocks
            layers_per_block: Number of layers per dense block
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(DenseNetHiggs, self).__init__()
        
        self.initial_dense = layers.Dense(
            initial_units, activation='relu', kernel_initializer='he_normal'
        )
        
        self.dense_blocks = []
        self.transition_layers = []
        
        current_units = initial_units
        for _ in range(num_dense_blocks):
            self.dense_blocks.append(
                DenseBlock(growth_rate, layers_per_block, dropout_rate, use_batch_norm)
            )
            
            # Transition layer (except after last block)
            current_units += growth_rate * layers_per_block
            transition_units = current_units // 2
            
            self.transition_layers.append(
                layers.Dense(transition_units, activation='relu', kernel_initializer='he_normal')
            )
            current_units = transition_units
        
        self.final_bn = layers.BatchNormalization()
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.output_layer = layers.Dense(
            1, activation='sigmoid', kernel_initializer='glorot_uniform'
        )
    
    def call(self, inputs, training=False):
        """Forward pass."""
        # Ensure 2D input
        if len(inputs.shape) == 2:
            x = tf.expand_dims(inputs, axis=-1)
        else:
            x = inputs
        
        x = self.initial_dense(x)
        
        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x, training=training)
            x = self.transition_layers[i](x)
        
        x = self.final_bn(x, training=training)
        x = self.global_avg_pool(x)
        
        return self.output_layer(x)
    
    def build_model(self) -> Model:
        """Build and return the Keras model."""
        inputs = layers.Input(shape=(self.input_dim,))
        
        if len(inputs.shape) == 2:
            x = tf.expand_dims(inputs, axis=-1)
        else:
            x = inputs
        
        x = self.initial_dense(x)
        
        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            x = self.transition_layers[i](x)
        
        x = self.final_bn(x)
        x = self.global_avg_pool(x)
        outputs = self.output_layer(x)
        
        return Model(inputs=inputs, outputs=outputs)


def create_model(
    model_type: str = 'mlp',
    input_dim: int = 28,
    **kwargs
) -> Model:
    """
    Factory function to create different model architectures.
    
    Args:
        model_type: Type of model ('mlp', 'resnet', 'densenet')
        input_dim: Number of input features
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Compiled Keras Model
    """
    if model_type == 'mlp':
        model = HiggsClassifier(input_dim=input_dim, **kwargs)
    elif model_type == 'resnet':
        model = ResNetHiggs(input_dim=input_dim, **kwargs)
    elif model_type == 'densenet':
        model = DenseNetHiggs(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.build_model()


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...")
    
    # Test MLP
    mlp = HiggsClassifier(input_dim=28, hidden_dims=[512, 256, 128])
    mlp_model = mlp.build_model()
    mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(f"MLP model created with {mlp_model.count_params():,} parameters")
    
    # Test ResNet
    resnet = ResNetHiggs(input_dim=28, num_blocks=3, units_per_block=128)
    resnet_model = resnet.build_model()
    resnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(f"ResNet model created with {resnet_model.count_params():,} parameters")
    
    # Test DenseNet
    densenet = DenseNetHiggs(input_dim=28, growth_rate=32, num_dense_blocks=2)
    densenet_model = densenet.build_model()
    densenet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(f"DenseNet model created with {densenet_model.count_params():,} parameters")
    
    # Test forward pass with dummy data
    import numpy as np
    dummy_input = np.random.randn(32, 28).astype(np.float32)
    
    mlp_output = mlp_model.predict(dummy_input, verbose=0)
    print(f"MLP output shape: {mlp_output.shape}")
    
    resnet_output = resnet_model.predict(dummy_input, verbose=0)
    print(f"ResNet output shape: {resnet_output.shape}")
    
    densenet_output = densenet_model.predict(dummy_input, verbose=0)
    print(f"DenseNet output shape: {densenet_output.shape}")
    
    print("\nAll models tested successfully!")
