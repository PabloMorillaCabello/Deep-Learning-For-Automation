"""
Task 2: Fashion MNIST with 100 hidden layers
Compare: sigmoid, ReLU, ELU, SELU activation functions
Analysis: Vanishing/Exploding Gradients Problem
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Load Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten images
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Class names for reference
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ============= Build Models with Different Activations =============
def build_deep_model(activation, use_batch_norm=False):
    """Build a 100-layer deep neural network"""
    model = keras.Sequential()

    # Input layer + first hidden layer
    model.add(layers.Dense(100, activation=activation, input_shape=(784,)))
    if use_batch_norm:
        model.add(layers.BatchNormalization())

    # 99 more hidden layers
    for i in range(99):
        model.add(layers.Dense(100, activation=activation))
        if use_batch_norm:
            model.add(layers.BatchNormalization())

    # Output layer
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Test different activation functions
activations = ['sigmoid', 'relu', 'elu', 'selu']
histories = {}
final_accuracies = {}

print("\n" + "="*60)
print("Training models with different activation functions")
print("="*60)

for activation in activations:
    print(f"\n--- Training with {activation.upper()} activation ---")
    model = build_deep_model(activation, use_batch_norm=False)

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    histories[activation] = history

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    final_accuracies[activation] = test_acc
    print(f"Test Accuracy with {activation}: {test_acc:.4f}")

# ============= Plot Training Curves Comparison =============
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, activation in enumerate(activations):
    ax = axes[idx]
    hist = histories[activation]

    ax.plot(hist.history['accuracy'], label='Training Accuracy', linewidth=2, marker='o', markersize=3)
    ax.plot(hist.history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'{activation.upper()} Activation - Accuracy Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add final accuracy to title
    final_acc = final_accuracies[activation]
    ax.text(0.5, 0.05, f'Final Test Acc: {final_acc:.4f}', 
            transform=ax.transAxes, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Fashion MNIST: 100-Layer DNN with Different Activations', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('task2_activation_comparison.png', dpi=150, bbox_inches='tight')
print("\nActivation comparison plot saved as 'task2_activation_comparison.png'")

# ============= Loss Comparison =============
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, activation in enumerate(activations):
    ax = axes[idx]
    hist = histories[activation]

    ax.plot(hist.history['loss'], label='Training Loss', linewidth=2, marker='o', markersize=3)
    ax.plot(hist.history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(f'{activation.upper()} Activation - Loss Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Fashion MNIST: Loss Curves Comparison', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('task2_loss_comparison.png', dpi=150, bbox_inches='tight')
print("Loss comparison plot saved as 'task2_loss_comparison.png'")

# ============= Summary Bar Plot =============
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = ax.bar(activations, [final_accuracies[a] for a in activations], color=colors, alpha=0.8, edgecolor='black', linewidth=2)

for bar, acc in zip(bars, [final_accuracies[a] for a in activations]):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.4f}\n({acc*100:.2f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Fashion MNIST: Final Test Accuracy by Activation Function', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('task2_accuracy_summary.png', dpi=150, bbox_inches='tight')
print("Summary plot saved as 'task2_accuracy_summary.png'")

# ============= Analysis Output =============
print("\n" + "="*60)
print("VANISHING/EXPLODING GRADIENTS ANALYSIS")
print("="*60)

print("""
Key Observations:

1. SIGMOID Activation:
   - Prone to VANISHING GRADIENTS
   - Output range [0,1], derivative max at 0.25
   - In deep networks: gradients → 0, weight updates → 0
   - Expected: Poor training, high loss plateauing

2. ReLU Activation:
   - Mitigates vanishing gradients for positive values
   - Risk of DEAD NEURONS (output 0 for x<0)
   - Better than sigmoid but still can suffer in very deep networks
   - Expected: Better training but possible dead neuron issues

3. ELU (Exponential Linear Unit):
   - Smooth for negative values: α(e^x - 1)
   - Better gradient flow than ReLU
   - Avoids dead neurons more effectively
   - Expected: Smoother training curves

4. SELU (Scaled ELU):
   - Self-normalizing activation
   - Maintains mean and variance through layers (WHEN using proper initialization)
   - Biologically inspired
   - Expected: Best training stability for very deep networks
""")

print(f"\nFinal Test Accuracies:")
for activation in activations:
    print(f"  {activation.upper():8s}: {final_accuracies[activation]:.4f}")

print("\n" + "="*60)
print("Task 2 Complete!")
print("="*60)
