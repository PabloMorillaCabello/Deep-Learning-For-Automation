# Deep Learning Lab - Tasks 1-3

Complete implementation of three deep learning exercises covering:
- Learning rate optimization
- Activation function analysis
- Batch normalization & optimizer comparison

## Files Included

### Python Code
- `task1_mnist.py` - MNIST with learning rate search (10 min runtime)
- `task2_fashion_mnist.py` - Fashion-MNIST 100-layer network (15 min runtime)
- `task3_cifar10.py` - CIFAR-10 with optimizers (30-45 min runtime)

### Report
- `lab_report_template.md` - Complete lab report template with all analysis
- This README with execution instructions

## Requirements

```bash
pip install tensorflow keras numpy matplotlib scikit-learn
```

**Recommended Versions:**
- TensorFlow >= 2.10
- Python 3.8+
- NumPy, Matplotlib

## Quick Start

### Step 1: Run Individual Tasks

```bash
# Task 1: MNIST Learning Rate Search (~10 minutes)
python task1_mnist.py

# Task 2: Fashion-MNIST Activation Functions (~15 minutes)
python task2_fashion_mnist.py

# Task 3: CIFAR-10 Optimizers (~30-45 minutes)
python task3_cifar10.py
```

### Step 2: Generate Plots

All scripts automatically generate plots:
- `task1_lr_search.png`
- `task1_training_curves.png`
- `task2_activation_comparison.png`
- `task2_loss_comparison.png`
- `task2_accuracy_summary.png`
- `task3_lr_search.png`
- `task3_batch_norm_comparison.png`
- `task3_optimizer_comparison.png`
- `task3_optimizer_summary.png`

### Step 3: Fill Report Template

1. Copy `lab_report_template.md`
2. Replace `[Your Name]`, `[Your value]` placeholders with actual results
3. Copy console output into report
4. Insert PNG plots into document
5. Export to PDF using Markdown editor or Pandoc:
   ```bash
   pandoc lab_report.md -o lab_report.pdf
   ```

## Task Descriptions

### Task 1: MNIST with Learning Rate Search
- **Goal**: Train MLP to >98% accuracy on MNIST
- **Key Steps**:
  1. Load MNIST dataset
  2. Perform exponential learning rate search (10^-4 to 10^0)
  3. Train with optimal LR for 50 epochs
  4. Generate training curves
  5. Evaluate on test set

**Output**: Learning rate search plot + training curves

**Expected Results**: 
- Test accuracy: >98%
- Optimal LR: ~0.001-0.01
- Training time: ~10 minutes

---

### Task 2: Fashion-MNIST 100-Layer Networks
- **Goal**: Compare vanishing gradients across 4 activation functions
- **Activation Functions**:
  1. Sigmoid - prone to vanishing gradients
  2. ReLU - mitigates issue but has dead neurons
  3. ELU - smoother gradient flow
  4. SELU - self-normalizing, best for deep networks

**Architecture**: 100 hidden layers × 100 neurons each

**Key Analysis**:
- Sigmoid: ~50% accuracy (severe vanishing gradients)
- ReLU: ~80-85% accuracy (good but imperfect)
- ELU: ~87-90% accuracy (good gradient flow)
- SELU: ~90-93% accuracy (best for deep networks)

**Output**: Activation comparison plots + loss curves + accuracy bars

---

### Task 3: CIFAR-10 Deep Learning
**Part A: Learning Rate Search**
- Quick 5-epoch search for optimal LR
- Tests 9 different learning rates
- Identifies optimal range

**Part B: Batch Normalization Comparison**
- Train 20-layer DNN without BN: baseline
- Train 20-layer DNN with BN: improved version
- Compare convergence speed, final accuracy, training stability

**Expected BN Impact**:
- Faster convergence: ~20-30% fewer epochs
- Higher accuracy: ~3-5% improvement
- Smoother loss curves

**Part C: Optimizer Comparison** (7 optimizers)
1. **SGD** - baseline, slowest
2. **Momentum SGD** - velocity accumulation
3. **Nesterov SGD** - look-ahead momentum
4. **AdaGrad** - adaptive per-parameter LR
5. **RMSProp** - adaptive without monotonic decay
6. **Adam** - momentum + adaptive (industry standard)
7. **Nadam** - Nesterov + Adam (best modern option)

**Convergence Speed Ranking**:
- Fastest: Nadam ≈ Adam >> RMSProp >> Nesterov ≈ Momentum >> SGD

**Output**: 
- Optimizer comparison (accuracy, loss)
- All training curves
- Summary bar charts

---

## Detailed Running Instructions

### Option 1: Run All Tasks (1-2 hours total)

```bash
# Sequential execution
python task1_mnist.py
python task2_fashion_mnist.py
python task3_cifar10.py
```

### Option 2: Run Individual Tasks

```bash
# Just Task 1
python task1_mnist.py

# Just Task 2
python task2_fashion_mnist.py

# Just Task 3
python task3_cifar10.py
```

### Option 3: Run Specific Code

Edit any .py file and comment out/remove early stopping or epochs to run shorter tests.

## Expected Outputs

### Console Output Example

```
Loading MNIST dataset...
Training set shape: (60000, 784)
Test set shape: (10000, 784)

==================================================
PHASE 1: Learning Rate Search
==================================================

Testing learning rates: [1.e-04 3.e-04 1.e-03 3.e-03 1.e-02 3.e-02 1.e-01 3.e-01 1.e+00]
This may take a few minutes...

Training with LR=0.000100... Val Loss: 0.3456
Training with LR=0.000316... Val Loss: 0.2345
Training with LR=0.001000... Val Loss: 0.1234  ← Best
...

Optimal Learning Rate: 0.001000

==================================================
PHASE 2: Training with Optimal Learning Rate
==================================================

Model: Sequential
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 256)              200960
batch_normalization (Batch)  (None, 256)              1024
...
Total params: 246,442
...

Epoch 1/50
1875/1875 [==============================] - 2s/batch - loss: 0.2567 - accuracy: 0.9234
...
Epoch 50/50
1875/1875 [==============================] - 2s/batch - loss: 0.0234 - accuracy: 0.9923

==================================================
PHASE 3: Evaluation
==================================================

Test Loss: 0.0456
Test Accuracy: 0.9876 (98.76%)

Learning rate search plot saved as 'task1_lr_search.png'
Training curves saved as 'task1_training_curves.png'

==================================================
Task 1 Complete!
==================================================
```

## Report Creation Steps

### Method 1: Using Microsoft Word / Google Docs
1. Copy markdown text from report template
2. Open in Pandoc or convert with online converter
3. Add plots as images
4. Format and export to PDF

### Method 2: Using Pandoc (Recommended)
```bash
# Install pandoc: https://pandoc.org/installing.html

# Convert to PDF
pandoc lab_report.md -o lab_report.pdf

# With styling
pandoc lab_report.md -o lab_report.pdf \
  --from markdown+yaml_metadata_block \
  --pdf-engine=xelatex \
  --variable mainfont="Calibri"
```

### Method 3: Using Jupyter Notebook
1. Copy content into Jupyter cells
2. Insert plots using `![](plot.png)`
3. Export notebook as PDF

### Method 4: Using Overleaf (LaTeX)
1. Create new project
2. Copy template (provided in Appendix)
3. Upload plots
4. Compile to PDF

## Troubleshooting

### Issue: "No module named 'tensorflow'"
**Solution**:
```bash
pip install tensorflow --upgrade
```

### Issue: Model training very slow
**Solution**:
- Reduce number of epochs in code
- Reduce dataset size for testing
- Use smaller batch sizes
- Check GPU availability: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### Issue: "Out of memory" error
**Solution**:
- Reduce batch size: change `batch_size=128` to `batch_size=64`
- Reduce model size: fewer layers/neurons
- Reduce dataset: use fewer samples for testing

### Issue: Plots not appearing
**Solution**:
```python
# Add to code:
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
```

## Key Concepts

### Learning Rate Search
- **Why**: Different datasets/architectures need different LRs
- **How**: Test 8-10 values exponentially spaced
- **Sweet Spot**: Balance between speed and stability
- **Red Flag**: Loss diverges (LR too high) or decreases (LR too low)

### Activation Functions
- **Sigmoid**: Old, suffers from vanishing gradients
- **ReLU**: Modern standard, simple but has issues
- **ELU/SELU**: Improved versions, better for deep networks
- **Selection**: Use SELU for very deep, ReLU for standard

### Batch Normalization
- **Purpose**: Standardize layer inputs to mean=0, var=1
- **Benefit**: Faster convergence, more stable training
- **Trade-off**: Slight computation overhead
- **Usage**: Add after Dense layers, before activation

### Optimizers
- **SGD**: Simple baseline, educational
- **Momentum/Nesterov**: Classical improvements
- **AdaGrad/RMSProp**: Adaptive learning rates
- **Adam/Nadam**: Modern, combine best of both worlds
- **Recommendation**: Use Adam or Nadam for most applications

## Performance Benchmarks (on typical hardware)

| Task | Runtime | GPU Required | Key Metric |
|------|---------|--------------|-----------|
| Task 1 | ~10 min | No | Test Acc >98% |
| Task 2 | ~15 min | No | SELU acc >90% |
| Task 3 | ~30-45 min | Ideal | Adam acc >55% |

## Important Notes

1. **First Run**: Initial imports may take time as TensorFlow builds graph
2. **Random Seeds**: Results vary slightly between runs (stochasticity)
3. **GPU**: Significantly faster if available, all code compatible
4. **Early Stopping**: Included to prevent overfitting and save time
5. **Memory**: Task 3 may need 4GB+ RAM; reduce batch size if issues

## Lab Session Presentation Tips

### What to Emphasize

**Task 1**: 
- Importance of learning rate selection
- Show the exponential search plot
- Explain why optimal LR prevents divergence

**Task 2**:
- Vanishing gradient problem in deep networks
- Why modern activations enable 100+ layers
- Show difference between sigmoid and SELU curves

**Task 3**:
- Batch normalization's speed-up and accuracy gain
- Convergence behavior differences between optimizers
- Why Adam/Nadam are industry standard

### Questions to Be Ready For

1. Why does sigmoid fail with 100 layers?
   - Gradients multiply through layers: 0.25^100 ≈ 0
   - Early layers don't update weights

2. What's batch normalization doing?
   - Normalizing layer inputs to mean=0, var=1
   - Reducing internal covariate shift

3. Why is Adam better than SGD?
   - Per-parameter adaptive learning rates
   - Combines momentum and RMSProp ideas
   - Less tuning needed

4. When would you use each optimizer?
   - SGD: Fine-tuning, research
   - Momentum/Nesterov: Classical approaches
   - RMSProp: RNNs, specific cases
   - Adam/Nadam: Default choice for 90% of cases

---

## Citation Guide

If using this code in papers/reports, cite as:

```bibtex
@misc{dllab2025,
  title={Deep Learning Lab: Neural Network Training & Optimization},
  author={[Your Name]},
  year={2025},
  note={Comprehensive implementation of learning rate search, activation function analysis, and optimizer comparison}
}
```

## License

Educational use only. Feel free to modify and extend for learning purposes.

---

**Last Updated**: December 2025  
**Tested With**: TensorFlow 2.13, Python 3.10  
**Total Execution Time**: ~2 hours for all tasks

Good luck with your lab! 🚀
