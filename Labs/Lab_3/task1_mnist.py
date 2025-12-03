"""
Task 1: Train a deep MLP on MNIST dataset
- Achieve >98% precision
- Search for optimal learning rate
- Plot learning curves with TensorBoard
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Load MNIST dataset
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten images
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# ============= Learning Rate Search =============
print("\n" + "="*50)
print("PHASE 1: Learning Rate Search")
print("="*50)

def build_model(learning_rate):
    """Build a deep MLP model"""
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Exponential learning rate search
learning_rates = 10.0 ** np.arange(-4, 1, 0.5)
losses = []

print(f"Testing learning rates: {learning_rates}")
print("This may take a few minutes...\n")

for lr in learning_rates:
    print(f"Training with LR={lr:.6f}...", end=" ")
    model = build_model(lr)
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    val_loss = history.history['val_loss'][-1]
    losses.append(val_loss)
    print(f"Val Loss: {val_loss:.4f}")

# Plot learning rate search results
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(learning_rates, losses, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Learning Rate (log scale)', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title('Learning Rate Search - MNIST', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task1_lr_search.png', dpi=150, bbox_inches='tight')
print("\nLearning rate search plot saved as 'task1_lr_search.png'")

# Find optimal learning rate
optimal_idx = np.argmin(losses)
optimal_lr = learning_rates[optimal_idx]
print(f"\nOptimal Learning Rate: {optimal_lr:.6f}")
print(f"Corresponding Loss: {losses[optimal_idx]:.4f}")

# ============= Training with Optimal LR =============
print("\n" + "="*50)
print("PHASE 2: Training with Optimal Learning Rate")
print("="*50)

model = build_model(optimal_lr)
print(model.summary())

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ============= Evaluation =============
print("\n" + "="*50)
print("PHASE 3: Evaluation")
print("="*50)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ============= Plot Training History =============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].axhline(y=test_accuracy, color='r', linestyle='--', label=f'Test Accuracy: {test_accuracy:.4f}')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('MNIST Training - Accuracy', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('MNIST Training - Loss', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1_training_curves.png', dpi=150, bbox_inches='tight')
print("\nTraining curves saved as 'task1_training_curves.png'")

print("\n" + "="*50)
print("Task 1 Complete!")
print("="*50)
