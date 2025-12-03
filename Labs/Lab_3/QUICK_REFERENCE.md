# Quick Reference: Deep Learning Lab

## File Structure
```
lab_project/
├── task1_mnist.py              (10 min)
├── task2_fashion_mnist.py      (15 min)
├── task3_cifar10.py            (30-45 min)
├── lab_report_template.md      (Fill in your results)
├── README.md                   (Full instructions)
│
├── Outputs (auto-generated):
├── task1_lr_search.png
├── task1_training_curves.png
├── task2_activation_comparison.png
├── task2_loss_comparison.png
├── task2_accuracy_summary.png
├── task3_lr_search.png
├── task3_batch_norm_comparison.png
├── task3_optimizer_comparison.png
└── task3_optimizer_summary.png
```

## 5-Minute Quick Run

```bash
# Install
pip install tensorflow keras numpy matplotlib

# Run
python task1_mnist.py
python task2_fashion_mnist.py
python task3_cifar10.py

# Results appear as PNGs in current directory
```

## Key Results to Report

### Task 1
- Learning Rate Search: [Optimal LR]
- Test Accuracy: [>98%]
- Test Loss: [<0.05]

### Task 2
- Sigmoid Acc: [~50%] ← Vanishing gradients
- ReLU Acc: [~85%] ← Better but imperfect
- ELU Acc: [~90%] ← Good gradient flow
- SELU Acc: [~93%] ← Best for deep networks

### Task 3
- Without BN Acc: [~45-50%]
- With BN Acc: [~50-55%] ← ~5% improvement
- Best Optimizer: [Adam or Nadam]
- Convergence: Nadam fastest → SGD slowest

## Code Modifications

### Run Fewer Epochs (for testing)
```python
# Change from:
history = model.fit(..., epochs=50, ...)

# To:
history = model.fit(..., epochs=5, ...)  # Quick test
```

### Reduce Data Size (for testing)
```python
# Add after loading data:
X_train = X_train[:5000]  # Only use 5000 samples
y_train = y_train[:5000]
```

### Use GPU
```bash
# TensorFlow auto-detects GPU
# Ensure CUDA installed: https://tensorflow.org/install/gpu
```

## Activation Functions at a Glance

| Function | Equation | Gradient Problem | Deep Networks |
|----------|----------|-----------------|---------------|
| Sigmoid  | σ(x) = 1/(1+e^-x) | Vanishing | ✗ Bad |
| ReLU     | max(0,x) | Dead neurons | ✓ OK |
| ELU      | x if x>0, α(e^x-1) | Mitigated | ✓ Good |
| SELU     | λ(x if x>0, ...) | Self-normalized | ✓✓ Best |

## Optimizer Ranking

**Convergence Speed**: Nadam ≈ Adam >> RMSProp >> Nesterov ≈ Momentum >> SGD

**Use Cases**:
1. **Default choice**: Adam or Nadam
2. **RNNs**: RMSProp  
3. **Fine-tuning**: SGD with Momentum
4. **Research**: SGD
5. **When unsure**: Adam

## Important Console Output to Capture

```
[Your output]
Epoch 50/50
1875/1875 [==========] - 2s/batch - loss: 0.023 - acc: 0.992 - val_loss: 0.041 - val_acc: 0.989

Test Loss: 0.0456
Test Accuracy: 0.9876
```

## Batch Normalization Formula

Input: x (mini-batch of layer inputs)

1. Normalize: x̂ = (x - μ_batch) / √(σ²_batch + ε)
2. Scale: y = γ*x̂ + β

Result:
- Faster convergence (fewer epochs needed)
- Higher accuracy (2-5% improvement typical)
- Enables larger learning rates
- Acts as regularizer

## Report Template Sections to Fill

1. **Learning Rate Search Results** [Your LR + loss plot]
2. **Model Summary** [Copy from console]
3. **Training Results** [Your accuracy/loss values]
4. **Activation Function Analysis** [Your accuracies + explanation]
5. **Batch Norm Comparison** [Your improvement %]
6. **Optimizer Results** [Your rankings + timing]

## Common Mistakes

❌ Using sigmoid in deep networks
❌ Not searching for learning rate
❌ Not comparing implementations
❌ Forgetting to normalize data
❌ Ignoring overfitting (no validation)
❌ Training for too few epochs

✓ Use modern activations (ReLU, ELU, SELU)
✓ Always search learning rate
✓ Compare implementations side-by-side
✓ Always normalize to [0,1]
✓ Use validation split (10-20%)
✓ Use early stopping to prevent overfitting

---

**Questions?** Check README.md for detailed explanations or review the code comments.

