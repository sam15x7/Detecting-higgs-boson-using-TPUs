# Higgs Boson Detection Architecture

## Overview

This document describes the neural network architectures implemented for Higgs Boson detection using TPUs.

## Problem Statement

The Higgs Boson detection problem is a binary classification task where we need to distinguish between:
- **Signal events**: Particle collisions that produce Higgs bosons
- **Background events**: Other particle collisions that mimic the signal

### Input Features (28 total)

The dataset contains 28 kinematic features measured by particle detectors:

1. **Lepton features** (3):
   - `lepton_pT`: Transverse momentum of the lepton
   - `lepton_eta`: Pseudorapidity of the lepton
   - `lepton_phi`: Azimuthal angle of the lepton

2. **Missing energy features** (2):
   - `missing_energy_magnitude`: Magnitude of missing transverse energy
   - `missing_energy_phi`: Azimuthal angle of missing energy

3. **Jet features** (20):
   - For each of up to 4 jets:
     - `jet_pt`: Transverse momentum
     - `jet_eta`: Pseudorapidity
     - `jet_phi`: Azimuthal angle
     - `jet_btag`: B-tagging discriminant

4. **Invariant mass features** (7):
   - `m_jj`: Invariant mass of two jets
   - `m_jjj`: Invariant mass of three jets
   - `m_lv`: Invariant mass of lepton and neutrino
   - `m_jlv`: Invariant mass of jet, lepton, and neutrino
   - `m_bb`: Invariant mass of two b-jets
   - `m_wbb`: Invariant mass of W boson and two b-jets
   - `m_wwbb`: Invariant mass of two W bosons and two b-jets

## Model Architectures

### 1. Multi-Layer Perceptron (MLP) - `HiggsClassifier`

The baseline architecture is a feedforward neural network with:

```
Input (28) → Dense(512) → BN → ReLU → Dropout → 
             Dense(256) → BN → ReLU → Dropout →
             Dense(128) → BN → ReLU → Dropout →
             Dense(1) → Sigmoid
```

**Key Features:**
- Configurable hidden layer dimensions
- Batch normalization for stable training
- Dropout for regularization
- He normal initialization for ReLU activations

### 2. ResNet-Style Architecture - `ResNetHiggs`

Implements residual connections to enable deeper networks:

```
Input (28) → Dense(128) → BN → ReLU → Dropout
                    ↓
             [Residual Block] × N
                    ↓
             Global Average Pooling
                    ↓
             Dense(1) → Sigmoid
```

**Residual Block:**
```
Input → Dense → BN → ReLU → Dropout → Dense → BN → (+) → ReLU → Output
        ↑___________________________________________|
```

**Benefits:**
- Mitigates vanishing gradient problem
- Enables training of deeper networks
- Identity skip connections preserve information flow

### 3. DenseNet-Style Architecture - `DenseNetHiggs`

Implements dense connections where each layer receives inputs from all preceding layers:

```
Input (28) → Dense(64)
                ↓
         [Dense Block] → Transition Layer
                ↓
         [Dense Block] → Transition Layer
                ↓
         Global Average Pooling
                ↓
         Dense(1) → Sigmoid
```

**Dense Block:**
Each layer concatenates its output with all previous feature maps:
```
Layer 1: F1 = H1(X0)
Layer 2: F2 = H2([X0, F1])
Layer 3: F3 = H3([X0, F1, F2])
...
```

**Benefits:**
- Maximum information flow between layers
- Feature reuse across the network
- Implicit deep supervision

## TPU Optimization Strategies

### 1. Mixed Precision Training

TPUs excel at float32 operations. We use:
- Float32 for all computations
- Automatic loss scaling when beneficial

### 2. Batch Size Optimization

For TPU efficiency:
- Use large batch sizes (1024-8192)
- Scale learning rate linearly with batch size
- Use gradient accumulation if needed

### 3. Data Pipeline Optimization

```python
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

- `drop_remainder=True`: Ensures static shapes for XLA compilation
- `prefetch`: Overlaps data loading with computation

### 4. Model Compilation

```python
with strategy.scope():
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', AMSMetric()]
    )
```

- All model creation within `strategy.scope()`
- Enables distributed training across TPU cores

## Evaluation Metrics

### Primary Metric: Approximate Median Significance (AMS)

The AMS metric approximates the statistical significance of a discovery:

```
AMS = √[2((s + b)ln(1 + s/b) - s)]

where:
- s = number of signal events selected
- b = number of background events selected
```

**Regularized version:**
```
b_reg = b + 1/br²

where br ≈ 0.1 is the expected signal/background ratio
```

### Secondary Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

## Training Strategy

### Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Learning Rate | 0.001 | Initial learning rate |
| Batch Size | 1024 | Samples per batch |
| Dropout Rate | 0.3 | Dropout probability |
| Epochs | 10-50 | Training epochs |
| Optimizer | Adam | Adaptive optimizer |

### Learning Rate Schedule

We use ReduceLROnPlateau:
- Monitor validation loss
- Reduce by factor of 0.5 on plateau
- Patience: 2-3 epochs
- Minimum LR: 1e-7

### Regularization

1. **Dropout**: Applied after each dense layer
2. **Batch Normalization**: Stabilizes training
3. **Early Stopping**: Prevents overfitting
4. **Weight Decay**: Optional L2 regularization

## Performance Benchmarks

### Expected Results (on full dataset)

| Model | Accuracy | AMS Score | Training Time (TPU v3) |
|-------|----------|-----------|------------------------|
| MLP | ~85% | ~3.40 | ~5 min/epoch |
| ResNet | ~86% | ~3.50 | ~7 min/epoch |
| DenseNet | ~87% | ~3.60 | ~8 min/epoch |

### TPU Speedup

Compared to single GPU:
- **Training**: 2-3x faster
- **Inference**: 3-5x faster
- **Scaling**: Near-linear with number of TPU cores

## Usage Example

```python
from src.model import HiggsClassifier
from src.train import Trainer
from src.data_loader import HiggsDataset

# Load data
dataset = HiggsDataset(data_path='data/')
train_loader, val_loader, _ = dataset.get_loaders(batch_size=1024)

# Create model
model = HiggsClassifier(
    input_dim=28,
    hidden_dims=[512, 256, 128],
    dropout_rate=0.3
).build_model()

# Train on TPU
trainer = Trainer(
    model=model,
    train_dataset=train_loader,
    val_dataset=val_loader,
    learning_rate=0.001,
    tpu_name=None  # Auto-detect
)

trainer.train(epochs=20)
```

## References

1. Baldi, P., et al. "Searching for exotic particles in high-energy physics with deep learning." Nature Communications 5 (2014).
2. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
3. Huang, G., et al. "Densely Connected Convolutional Networks." CVPR 2017.
4. ATLAS Collaboration. "Measurement of the Higgs boson mass." Physical Review Letters (2015).
