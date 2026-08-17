# Optimizer Visualizer: From SGD to AdamW

🔗 **Live Demo:** https://sg721642-deep-learning-optimizer-visualizer-app-mnapjy.streamlit.app/

An interactive visual tool to explore how deep learning optimizers learn from first principles. Implemented completely from scratch in pure NumPy, featuring 2D loss surface simulations and real-time Multi-Layer Perceptron benchmarking.

---

## Overview

- **7 From-Scratch Optimizers:** SGD, SGD with Momentum, NAG (Nesterov Accelerated Gradient), AdaGrad, RMSProp, Adam, and AdamW.
- **Pure NumPy Core:** 3-layer neural network (`30 → 16 → 8 → 1`) with He/Xavier initialization, binary cross-entropy loss, and analytical backpropagation.
- **Part A (2D Playground):** 4 loss surfaces ($L_1$ to $L_4$) with condition numbers $\kappa \in [10, 1000]$, synchronized 2D contour map and loss curves, and interactive animation controls.
- **Part B (Neural Network):** Live training dashboard on the Breast Cancer Wisconsin dataset with real-time loss, accuracy, and effective learning rate ($\eta_{\text{eff}}$) tracking, plus an auto-generated comparison table.
- **Restrictions Compliance:** 0% `torch.optim`, 0% `keras.optimizers`, 0% autograd engines.

---

## Optimizer Update Rules (Pure NumPy)

| Optimizer | Mathematical Update Rule | Primary Mechanism |
|---|---|---|
| **1. SGD** | $\theta_{t+1} = \theta_t - \eta g_t$ | Direct first-order gradient descent |
| **2. Momentum** | $v_t = \beta v_{t-1} + (1-\beta)g_t, \quad \theta_{t+1} = \theta_t - \eta v_t$ | Velocity buffer dampens high-frequency oscillation |
| **3. NAG** | $g_{la} = \nabla L(\theta_t - \beta v_{t-1}), \quad v_t = \beta v_{t-1} + (1-\beta)g_{la}, \quad \theta_{t+1} = \theta_t - \eta v_t$ | Look-ahead gradient reduces overshoot |
| **4. AdaGrad** | $G_t = G_{t-1} + g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t+\epsilon}} \odot g_t$ | Parameter-wise adaptive learning rate |
| **5. RMSProp** | $v_t = \beta v_{t-1} + (1-\beta)g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t+\epsilon}} \odot g_t$ | Exponential moving average of squared gradients |
| **6. Adam** | $\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \ \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \hat{m}_t$ | Combines first and second moments with bias correction |
| **7. AdamW** | $\theta_{t+1} = \theta_t (1 - \eta \lambda) - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$ | Decoupled weight decay for proper regularization |

---

## Project Structure

```text
Deep-Learning-Optimizer-Visualizer/
├── app.py                     # Streamlit application entry point
├── src/
│   ├── __init__.py            # Package exports
│   ├── optimizers.py          # 7 pure NumPy optimizers
│   ├── surfaces.py            # 2D loss surfaces (L1-L4), analytical gradients, Hessians
│   ├── neural_net.py          # BinaryMLP (Dense, ReLU, Sigmoid, BCE loss, backpropagation)
│   ├── dataset.py             # DatasetManager (Breast Cancer dataset, split, standardization)
│   ├── visualization.py       # Plotly charts (Contour map, loss curves, training dashboard)
│   └── experiment.py          # Training engine, comparison table & convergence epoch calculation
├── tests/
│   ├── test_optimizers.py     # Unit tests for all 7 optimizers
│   ├── test_surfaces.py       # Unit tests for 2D surfaces and analytical gradients
│   ├── test_neural_net.py     # Forward pass and numerical gradient check
│   ├── test_streamlit_app.py  # Streamlit UI AppTest integration test
│   └── test_restrictions.py   # Static analysis verifying 0 prohibited imports
├── requirements.txt           # Dependency list
├── .gitignore                 # Excludes cache, virtual environments, sensitive files
├── REQUIREMENTS_CHECKLIST.md  # Comprehensive compliance matrix
├── REFLECTION_ANSWERS.md      # Answers to Sections A7 and B4
├── CONCLUSION.md              # Academic conclusion on optimizer evolution
└── README.md                  # Project documentation
```

---

## Installation & Usage

### 1. Setup Environment
```bash
git clone https://github.com/sg721642/Deep-Learning-Optimizer-Visualizer.git
cd Deep-Learning-Optimizer-Visualizer
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python3 -m pytest tests/
```

### 3. Launch Application
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## Benchmark Results (Breast Cancer Dataset)

*Generated via `src/experiment.py` (80 Epochs, Batch Size 32, Seed 42):*

| Optimizer | Final Train Loss | Final Test Loss | Train Acc. | Test Acc. | Convergence Epoch (≤ 1% Final Loss) |
|---|---|---|---|---|---|
| **SGD** | 0.0657 | 0.1085 | 98.90% | 95.61% | Epoch 74 |
| **Momentum** | 0.0660 | 0.1091 | 98.68% | 95.61% | Epoch 74 |
| **NAG** | 0.0653 | 0.1085 | 98.90% | 96.49% | Epoch 76 |
| **AdaGrad** | 0.0321 | 0.0790 | 99.34% | 96.49% | Epoch 78 |
| **RMSProp** | 0.0004 | 0.4696 | 100.00% | 95.61% | Epoch 80 |
| **Adam** | 0.0062 | 0.2005 | 100.00% | 96.49% | Epoch 80 |
| **AdamW** | 0.0010 | 0.1406 | 100.00% | 96.49% | Epoch 77 |

---

## Restrictions Compliance

- **No built-in optimizers:** `torch.optim` and `keras.optimizers` are absent from the entire codebase (verified by `tests/test_restrictions.py`).
- **No automatic differentiation:** All gradients and backpropagation rules are derived and implemented analytically in NumPy.
- **Unified implementations:** The same optimizer classes from `src/optimizers.py` are reused in both Part A and Part B.
