"""
Task 3: Train deep neural network on CIFAR-10
- Deep architecture: 20 hidden layers × 100 neurons
- He initialization + ELU activation
- Learning rate search
- Batch Normalization comparison
- Compare optimizers: SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam, Nadam
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, initializers
import tensorflow as tf

# Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten images
X_train_flat = X_train.reshape(-1, 32*32*3)
X_test_flat = X_test.reshape(-1, 32*32*3)

print(f"Training set shape: {X_train_flat.shape}")
print(f"Test set shape: {X_test_flat.shape}")

# Flatten y
y_train = y_train.flatten()
y_test = y_test.flatten()

# ============= PHASE 1: Learning Rate Search =============
print("\n" + "="*70)
print("PHASE 1: Learning Rate Search (with ELU + He Initialization)")
print("="*70)

def build_cifar10_model(learning_rate, use_batch_norm=False):
    """Build 20-layer DNN for CIFAR-10 with He initialization"""
    model = keras.Sequential()

    # Input + first hidden layer
    model.add(layers.Dense(100, activation='elu', 
                          kernel_initializer=initializers.HeNormal(),
                          input_shape=(3072,)))
    if use_batch_norm:
        model.add(layers.BatchNormalization())

    # 19 more hidden layers
    for i in range(19):
        model.add(layers.Dense(100, activation='elu',
                              kernel_initializer=initializers.HeNormal()))
        if use_batch_norm:
            model.add(layers.BatchNormalization())

    # Output layer
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Quick learning rate search
learning_rates = 10.0 ** np.arange(-4, -1, 0.5)
lr_losses = []

print(f"\nTesting learning rates: {learning_rates}")
print("(Training for 5 epochs per LR for quick search)\n")

for lr in learning_rates:
    print(f"LR={lr:.6f}...", end=" ")
    model = build_cifar10_model(lr, use_batch_norm=False)
    history = model.fit(
        X_train_flat, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )
    val_loss = history.history['val_loss'][-1]
    lr_losses.append(val_loss)
    print(f"Val Loss: {val_loss:.4f}")

# Plot LR search
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(learning_rates, lr_losses, 'ro-', linewidth=2, markersize=8)
ax.set_xlabel('Learning Rate (log scale)', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title('Learning Rate Search - CIFAR-10', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task3_lr_search.png', dpi=150, bbox_inches='tight')
print("\nLearning rate search plot saved")

optimal_lr = learning_rates[np.argmin(lr_losses)]
print(f"Optimal Learning Rate: {optimal_lr:.6f}")

# ============= PHASE 2: WITHOUT Batch Normalization =============
print("\n" + "="*70)
print("PHASE 2: Training WITHOUT Batch Normalization")
print("="*70)

model_no_bn = build_cifar10_model(optimal_lr, use_batch_norm=False)
print(model_no_bn.summary())

print("\nTraining (this may take 1-2 minutes)...")
history_no_bn = model_no_bn.fit(
    X_train_flat, y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

test_loss_no_bn, test_acc_no_bn = model_no_bn.evaluate(X_test_flat, y_test, verbose=0)
print(f"\nTest Loss (no BN): {test_loss_no_bn:.4f}")
print(f"Test Accuracy (no BN): {test_acc_no_bn:.4f}")

# ============= PHASE 3: WITH Batch Normalization =============
print("\n" + "="*70)
print("PHASE 3: Training WITH Batch Normalization")
print("="*70)

model_with_bn = build_cifar10_model(optimal_lr, use_batch_norm=True)
print(model_with_bn.summary())

print("\nTraining (this may take 1-2 minutes)...")
history_with_bn = model_with_bn.fit(
    X_train_flat, y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

test_loss_with_bn, test_acc_with_bn = model_with_bn.evaluate(X_test_flat, y_test, verbose=0)
print(f"\nTest Loss (with BN): {test_loss_with_bn:.4f}")
print(f"Test Accuracy (with BN): {test_acc_with_bn:.4f}")

# ============= Compare Batch Normalization =============
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy comparison
axes[0].plot(history_no_bn.history['accuracy'], label='No BN - Training', linewidth=2, alpha=0.7)
axes[0].plot(history_no_bn.history['val_accuracy'], label='No BN - Validation', linewidth=2, alpha=0.7)
axes[0].plot(history_with_bn.history['accuracy'], label='With BN - Training', linewidth=2, alpha=0.7)
axes[0].plot(history_with_bn.history['val_accuracy'], label='With BN - Validation', linewidth=2, alpha=0.7)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Batch Normalization Impact on Accuracy', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Loss comparison
axes[1].plot(history_no_bn.history['loss'], label='No BN - Training', linewidth=2, alpha=0.7)
axes[1].plot(history_no_bn.history['val_loss'], label='No BN - Validation', linewidth=2, alpha=0.7)
axes[1].plot(history_with_bn.history['loss'], label='With BN - Training', linewidth=2, alpha=0.7)
axes[1].plot(history_with_bn.history['val_loss'], label='With BN - Validation', linewidth=2, alpha=0.7)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Batch Normalization Impact on Loss', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task3_batch_norm_comparison.png', dpi=150, bbox_inches='tight')
print("\nBatch normalization comparison saved")

# ============= PHASE 4: Optimizer Comparison =============
print("\n" + "="*70)
print("PHASE 4: Optimizer Comparison")
print("="*70)

optimizers_dict = {
    'SGD': keras.optimizers.SGD(learning_rate=optimal_lr),
    'Momentum': keras.optimizers.SGD(learning_rate=optimal_lr, momentum=0.9),
    'Nesterov': keras.optimizers.SGD(learning_rate=optimal_lr, momentum=0.9, nesterov=True),
    'AdaGrad': keras.optimizers.Adagrad(learning_rate=optimal_lr),
    'RMSProp': keras.optimizers.RMSprop(learning_rate=optimal_lr),
    'Adam': keras.optimizers.Adam(learning_rate=optimal_lr),
    'Nadam': keras.optimizers.Nadam(learning_rate=optimal_lr)
}

optimizer_histories = {}
optimizer_results = {}

for opt_name, optimizer in optimizers_dict.items():
    print(f"\n--- Training with {opt_name} ---")

    model = keras.Sequential([
        layers.Dense(100, activation='elu', kernel_initializer=initializers.HeNormal(), input_shape=(3072,)),
        layers.BatchNormalization(),
    ])

    for i in range(19):
        model.add(layers.Dense(100, activation='elu', kernel_initializer=initializers.HeNormal()))
        model.add(layers.BatchNormalization())

    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train_flat, y_train,
        epochs=30,
        batch_size=128,
        validation_split=0.1,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )

    test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
    optimizer_histories[opt_name] = history
    optimizer_results[opt_name] = {'loss': test_loss, 'accuracy': test_acc, 'epochs': len(history.history['loss'])}

    print(f"Test Accuracy: {test_acc:.4f} | Epochs trained: {len(history.history['loss'])}")

# Plot optimizer comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Training accuracy
ax = axes[0, 0]
for opt_name, history in optimizer_histories.items():
    ax.plot(history.history['accuracy'], label=opt_name, linewidth=2, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Training Accuracy', fontsize=11)
ax.set_title('Optimizer Comparison - Training Accuracy', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Validation accuracy
ax = axes[0, 1]
for opt_name, history in optimizer_histories.items():
    ax.plot(history.history['val_accuracy'], label=opt_name, linewidth=2, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Validation Accuracy', fontsize=11)
ax.set_title('Optimizer Comparison - Validation Accuracy', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: Training loss
ax = axes[1, 0]
for opt_name, history in optimizer_histories.items():
    ax.plot(history.history['loss'], label=opt_name, linewidth=2, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Training Loss', fontsize=11)
ax.set_title('Optimizer Comparison - Training Loss', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: Validation loss
ax = axes[1, 1]
for opt_name, history in optimizer_histories.items():
    ax.plot(history.history['val_loss'], label=opt_name, linewidth=2, alpha=0.8)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Validation Loss', fontsize=11)
ax.set_title('Optimizer Comparison - Validation Loss', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('CIFAR-10: Optimizer Comparison (with Batch Normalization)', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('task3_optimizer_comparison.png', dpi=150, bbox_inches='tight')
print("\nOptimizer comparison plot saved")

# Summary bar plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

opt_names = list(optimizer_results.keys())
accuracies = [optimizer_results[n]['accuracy'] for n in opt_names]
losses = [optimizer_results[n]['loss'] for n in opt_names]

# Accuracy
colors = plt.cm.viridis(np.linspace(0, 1, len(opt_names)))
bars1 = axes[0].bar(opt_names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Test Accuracy', fontsize=12)
axes[0].set_title('Final Test Accuracy by Optimizer', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([0, 1.0])
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Loss
bars2 = axes[1].bar(opt_names, losses, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
for bar, loss in zip(bars2, losses):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Test Loss', fontsize=12)
axes[1].set_title('Final Test Loss by Optimizer', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('task3_optimizer_summary.png', dpi=150, bbox_inches='tight')
print("Optimizer summary plot saved")

# ============= Summary Analysis =============
print("\n" + "="*70)
print("SUMMARY: OPTIMIZER ANALYSIS")
print("="*70)

print("""
KEY DIFFERENCES BETWEEN OPTIMIZERS:

1. SGD (Stochastic Gradient Descent):
   - Simple, slow convergence
   - No adaptive learning rate
   - May get stuck in local minima
   - Expected: Lowest accuracy

2. Momentum:
   - Accumulates gradient velocity
   - Helps escape local minima
   - Faster convergence than vanilla SGD
   - Expected: Faster training, better accuracy

3. Nesterov Accelerated Gradient:
   - Looks ahead before applying gradient
   - More responsive than standard momentum
   - Theoretical convergence guarantees
   - Expected: Similar to Momentum, slightly better

4. AdaGrad:
   - Adaptive learning rate per parameter
   - Large updates for sparse features
   - Learning rate monotonically decreasing
   - Expected: Can suffer on long training (LR becomes too small)

5. RMSProp:
   - Improves on AdaGrad
   - Uses exponential moving average of squared gradients
   - Adaptive learning rate without monotonic decrease
   - Expected: Good convergence, faster than SGD/Momentum

6. Adam (Adaptive Moment Estimation):
   - Combines momentum + RMSProp ideas
   - Adaptive learning rates for each parameter
   - Generally very robust
   - Expected: Fast, stable convergence, good accuracy

7. Nadam (Nesterov + Adam):
   - Adam with Nesterov momentum instead of standard momentum
   - Best of both worlds: Nesterov's lookahead + Adam's adaptivity
   - Most advanced option
   - Expected: Potentially best performance
""")

print("\nTest Results Summary:")
print(f"{'Optimizer':<12} {'Accuracy':<12} {'Loss':<12} {'Epochs':<10}")
print("-" * 46)
for opt_name in opt_names:
    acc = optimizer_results[opt_name]['accuracy']
    loss = optimizer_results[opt_name]['loss']
    epochs = optimizer_results[opt_name]['epochs']
    print(f"{opt_name:<12} {acc:<12.4f} {loss:<12.4f} {epochs:<10}")

print("\n" + "="*70)
print("Task 3 Complete!")
print("="*70)
