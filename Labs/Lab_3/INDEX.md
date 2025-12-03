# 📚 DEEP LEARNING LAB - COMPLETE INDEX

## Quick Navigation

### 🚀 Getting Started (Read First!)
1. **START HERE**: Read `SUMMARY.txt` for complete overview
2. **Installation**: Follow `INSTALLATION.md` to set up environment
3. **Quick Start**: See `QUICK_REFERENCE.md` for 5-minute summary

### 💻 Run the Code
Execute tasks in order:
```bash
python task1_mnist.py          # ~10 min
python task2_fashion_mnist.py  # ~15 min
python task3_cifar10.py        # ~30-45 min
```

### 📝 Create Your Report
1. Open `lab_report_template.md`
2. Fill in all [Your value] placeholders with actual results
3. Insert generated PNG plots
4. Export to PDF: `pandoc lab_report_template.md -o lab_report.pdf`

### 🎓 Lab Session Presentation
1. Show `lab_report.pdf` to teacher
2. Walk through code in each `.py` file
3. Discuss results and explain what changed and why

---

## 📂 File Directory

```
lab_project/
│
├─ 🐍 PYTHON CODE (Executable)
│  ├─ task1_mnist.py              ← Run this first (~10 min)
│  │  └─ Outputs: 2 plots
│  ├─ task2_fashion_mnist.py      ← Run this second (~15 min)
│  │  └─ Outputs: 3 plots
│  └─ task3_cifar10.py            ← Run this last (~30-45 min)
│     └─ Outputs: 4 plots
│
├─ 📋 DOCUMENTATION (Read These)
│  ├─ INDEX.md                    ← This file
│  ├─ SUMMARY.txt                 ← Complete overview (start here!)
│  ├─ README.md                   ← Full detailed guide
│  ├─ QUICK_REFERENCE.md          ← Cheat sheet
│  └─ INSTALLATION.md             ← Setup instructions
│
├─ 📄 REPORT TEMPLATE
│  └─ lab_report_template.md      ← Fill in your results here
│
└─ 🖼️ GENERATED PLOTS (Appear After Running)
   ├─ task1_lr_search.png
   ├─ task1_training_curves.png
   ├─ task2_activation_comparison.png
   ├─ task2_loss_comparison.png
   ├─ task2_accuracy_summary.png
   ├─ task3_lr_search.png
   ├─ task3_batch_norm_comparison.png
   ├─ task3_optimizer_comparison.png
   └─ task3_optimizer_summary.png
```

---

## 📖 Document Guide

### SUMMARY.txt
**What**: Complete overview of entire lab
**Read Time**: 15-20 minutes
**Contains**:
- Executive summary of all 3 tasks
- Expected results and timings
- Presentation tips
- Troubleshooting guide
- Key formulas

**Best For**: Getting complete picture before starting

### README.md
**What**: Comprehensive instructions and guide
**Read Time**: 10-15 minutes
**Contains**:
- Detailed task descriptions
- Step-by-step execution instructions
- Expected outputs
- Performance benchmarks
- Common mistakes to avoid

**Best For**: Detailed understanding of each task

### QUICK_REFERENCE.md
**What**: Quick cheat sheet
**Read Time**: 5 minutes
**Contains**:
- File structure overview
- 5-minute quick run
- Key results to report
- Optimizer rankings
- Activation function comparison

**Best For**: Quick lookup during coding

### INSTALLATION.md
**What**: Setup and environment configuration
**Read Time**: 5-10 minutes
**Contains**:
- Installation instructions for Windows/Mac/Linux
- GPU setup (optional)
- Verification scripts
- Troubleshooting
- Dependency versions

**Best For**: Getting environment ready

### lab_report_template.md
**What**: Professional lab report template
**Length**: ~12 pages
**Contains**:
- Sections for all 3 tasks
- Placeholders for your results
- Analysis sections
- Theoretical background
- References section

**Best For**: Submitting your work

---

## 🎯 Task Summary

### Task 1: MNIST Learning Rate Search ⚙️

**Time**: ~10 minutes  
**Difficulty**: Easy  
**Concept**: Learning rate optimization

**What You Do**:
1. Load MNIST dataset
2. Test 9 different learning rates
3. Train with optimal LR
4. Achieve >98% accuracy

**What You Learn**:
- Importance of learning rate selection
- How exponential search finds optimal hyperparameter
- Impact on convergence and final accuracy

**Output**:
- `task1_lr_search.png` - Shows optimal LR
- `task1_training_curves.png` - Training curves

---

### Task 2: Fashion-MNIST Activation Functions 🔄

**Time**: ~15 minutes  
**Difficulty**: Medium  
**Concept**: Vanishing gradients problem

**What You Do**:
1. Build 100-layer deep network (intentionally deep!)
2. Test 4 activation functions: sigmoid, ReLU, ELU, SELU
3. Compare accuracy and loss curves
4. Analyze gradient flow in deep networks

**What You Learn**:
- Vanishing gradient problem: gradient decay ∝ 0.25^100
- Why sigmoid/tanh fail in deep networks
- Why modern activations (SELU, ELU) are necessary
- How activation functions determine network depth

**Expected Results**:
- Sigmoid: ~50% (fails!)
- ReLU: ~85% (okay)
- ELU: ~90% (good)
- SELU: ~93% (best)

**Output**:
- `task2_activation_comparison.png` - Accuracy curves
- `task2_loss_comparison.png` - Loss curves
- `task2_accuracy_summary.png` - Accuracy comparison

---

### Task 3: CIFAR-10 Batch Norm & Optimizers 🚀

**Time**: ~30-45 minutes  
**Difficulty**: Hard  
**Concepts**: Batch normalization, optimizer comparison

**Part A: Batch Normalization Impact**
1. Train 20-layer network WITHOUT batch norm
2. Train same network WITH batch norm
3. Compare convergence speed, accuracy, stability

Expected: 5-10% accuracy improvement, 20-30% fewer epochs

**Part B: Optimizer Comparison**
1. Test 7 different optimizers:
   - SGD (baseline, slowest)
   - Momentum SGD
   - Nesterov SGD
   - AdaGrad
   - RMSProp
   - Adam (industry standard)
   - Nadam (best modern option)
2. Compare convergence speed and final accuracy

Expected Ranking:
1. Nadam ≈ Adam (fastest, ~15 epochs)
2. RMSProp (fast, ~20 epochs)
3. Nesterov ≈ Momentum (moderate, ~25 epochs)
4. SGD (slow, ~30+ epochs)

**What You Learn**:
- How batch normalization stabilizes training
- Per-parameter adaptive learning rates
- Why modern optimizers (Adam, Nadam) are industry standard
- When to use which optimizer

**Output**:
- `task3_lr_search.png` - CIFAR-10 LR search
- `task3_batch_norm_comparison.png` - Batch norm impact
- `task3_optimizer_comparison.png` - Optimizer curves
- `task3_optimizer_summary.png` - Final results

---

## ⏱️ Timeline

| Task | Duration | GPU | Epochs | Output |
|------|----------|-----|--------|--------|
| Task 1 | 10 min | No | 50 | 2 plots |
| Task 2 | 15 min | No | 20 | 3 plots |
| Task 3 | 30-45 min | Yes* | 30+ | 4 plots |
| **Total** | **~2 hours** | Optional | - | **9 plots** |

*GPU recommended but not required for Task 3

---

## 📊 Generated Plots Reference

### Task 1 Plots

**task1_lr_search.png**
- X-axis: Learning Rate (log scale)
- Y-axis: Validation Loss
- Purpose: Show optimal LR (minimum point)

**task1_training_curves.png**
- Left: Accuracy over epochs
- Right: Loss over epochs
- Purpose: Show model convergence

### Task 2 Plots

**task2_activation_comparison.png**
- 4 subplots for sigmoid, relu, elu, selu
- Shows accuracy curves over epochs
- Purpose: Compare activation performance

**task2_loss_comparison.png**
- 4 subplots for sigmoid, relu, elu, selu
- Shows loss curves over epochs
- Purpose: Show training dynamics

**task2_accuracy_summary.png**
- Bar chart of final test accuracy
- Purpose: Clear comparison of activation functions

### Task 3 Plots

**task3_batch_norm_comparison.png**
- Left: Accuracy with/without BN
- Right: Loss with/without BN
- Purpose: Show BN impact on convergence

**task3_optimizer_comparison.png**
- 4 subplots (train acc, val acc, train loss, val loss)
- 7 lines (one per optimizer)
- Purpose: Compare all optimizers side-by-side

**task3_optimizer_summary.png**
- Left: Final accuracy bar chart
- Right: Final loss bar chart
- Purpose: Clear optimizer ranking

---

## 🎓 Learning Progression

The tasks are designed to build understanding progressively:

```
Task 1: Learn
  └─ How learning rates affect training
  └─ Basic hyperparameter optimization

       ↓ (Reinforce concepts)

Task 2: Understand
  └─ Why deep networks are hard (vanishing gradients)
  └─ How activations solve problems
  └─ Design principles matter

       ↓ (Apply concepts)

Task 3: Master
  └─ Batch normalization benefits
  └─ Optimizer selection impact
  └─ Modern deep learning best practices
```

---

## 💡 Key Concepts

### Learning Rate
- Controls step size for weight updates
- Too small: training too slow or stops
- Too large: loss diverges
- Optimal: usually 0.001 - 0.1

### Vanishing Gradients
- Problem: gradients exponentially decay through layers
- Cause: sigmoid derivative max = 0.25
- Math: gradient ∝ 0.25^100 ≈ 10^-60 with 100 layers
- Solution: ReLU, ELU, SELU activations

### Batch Normalization
- Normalizes layer inputs to mean=0, variance=1
- Prevents internal covariate shift
- Enables faster training and larger learning rates
- Trade-off: small computation overhead

### Optimizers
- **SGD**: Simple but slow
- **Momentum**: Better convergence (0.9x velocity accumulation)
- **Nesterov**: Look-ahead variant of momentum
- **Adam**: Adaptive learning rates + momentum (best default)
- **Nadam**: Adam + Nesterov (theoretically best)

---

## ✅ Execution Checklist

### Before Starting
- [ ] Python 3.8+ installed
- [ ] pip install tensorflow keras numpy matplotlib
- [ ] 4+ GB RAM available
- [ ] 2+ GB disk space free

### During Execution
- [ ] Task 1 completes: >98% accuracy achieved
- [ ] Task 1 plots created successfully
- [ ] Task 2 completes: SELU > ELU > ReLU > Sigmoid trend shown
- [ ] Task 2 plots created successfully
- [ ] Task 3 completes: Batch norm speedup visible
- [ ] Task 3 plots created successfully
- [ ] All 9 plots have correct labels and legends

### After Execution
- [ ] Console output saved/screenshotted
- [ ] All 9 plots copied to report
- [ ] Report template filled with results
- [ ] [Your value] placeholders replaced
- [ ] Report exported to PDF
- [ ] Code presented with clear comments
- [ ] Ready to discuss results in lab session

---

## 🔗 File Dependencies

```
Code Files (Independent):
  task1_mnist.py ────→ Generates task1_*.png
  task2_fashion_mnist.py ──→ Generates task2_*.png
  task3_cifar10.py ──→ Generates task3_*.png

Report Template:
  lab_report_template.md ←─── Needs all 9 .png files
                               + console output
                               + your analysis

Documentation (Reference):
  SUMMARY.txt
  README.md
  QUICK_REFERENCE.md
  INSTALLATION.md
```

---

## 📞 Support & Troubleshooting

### Common Issues

**Issue**: "No module named 'tensorflow'"
- Solution: `pip install tensorflow --upgrade`

**Issue**: GPU not detected (slow training)
- Solution: Install CUDA/cuDNN or accept CPU training

**Issue**: Out of memory error
- Solution: Reduce batch_size in code (128 → 64)

**Issue**: Plots not displaying
- Solution: They auto-save as PNG (no display needed)

**Issue**: Different results than expected
- Solution: Normal variation due to randomness; same trends appear

### Getting Help

1. Check `TROUBLESHOOTING` section in README.md
2. Review `QUICK_REFERENCE.md` for concepts
3. Look at code comments in `.py` files
4. Consult official TensorFlow docs: https://tensorflow.org/

---

## 🎓 For Lab Session

**What to Bring**:
1. Completed `lab_report.pdf`
2. All `.py` code files
3. Generated PNG plots (or PDF with plots embedded)

**What to Be Ready to Discuss**:
1. Why learning rate matters (Task 1)
2. Vanishing gradient problem (Task 2)
3. How batch normalization helps (Task 3)
4. Why Adam/Nadam > SGD (Task 3)
5. Trade-offs of each technique

**Presentation Flow**:
1. Show report
2. Walk through code
3. Discuss plots and observations
4. Explain what changed and why
5. Answer teacher questions

---

## ⚡ Quick Command Reference

```bash
# Setup
pip install tensorflow keras numpy matplotlib

# Run all tasks
python task1_mnist.py
python task2_fashion_mnist.py
python task3_cifar10.py

# Convert report to PDF
pandoc lab_report_template.md -o lab_report.pdf

# View a plot (optional)
python -c "from PIL import Image; Image.open('task1_lr_search.png').show()"
```

---

## 📚 Additional Resources

- TensorFlow Official: https://www.tensorflow.org/
- Keras API: https://keras.io/
- Deep Learning Papers:
  - Batch Normalization: https://arxiv.org/abs/1502.03167
  - Adam Optimizer: https://arxiv.org/abs/1412.6980
  - Understanding Deep Networks: https://arxiv.org/abs/1509.02287

---

## 🎉 Final Words

You now have **EVERYTHING** needed to complete this lab:
- ✅ 3 complete Python implementations
- ✅ Professional report template
- ✅ Comprehensive documentation
- ✅ Quick reference guides
- ✅ Troubleshooting tips
- ✅ Presentation guidance

**Timeline**: ~2 hours of execution + 30 min report writing = 2.5 hours total

**Success Criteria**:
✓ All tasks run without errors
✓ Expected accuracies achieved
✓ All plots generated
✓ Professional report submitted
✓ Code and analysis presented

Good luck! You've got this! 🚀

---

**Document Version**: 1.0  
**Last Updated**: December 2025  
**Status**: Ready for submission  
