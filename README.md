# ⚡ From SGD to AdamW — Deep Learning Optimizer Visualizer

> **Building an Interactive Visual Tool to See How Optimizers Learn**  
> *A 100% from-scratch NumPy implementation of 7 fundamental deep learning optimizers, interactive 2D loss surface simulations, synchronized animations, and real-time Multi-Layer Perceptron neural network benchmarking.*

---

## 🌟 Key Highlights & Features

- **Pure NumPy Implementation from Scratch:**
  - **7 Stateful Optimizers:** SGD, SGD with Momentum, NAG (Nesterov Accelerated Gradient), AdaGrad, RMSProp, Adam, and AdamW.
  - **Neural Network Core:** 3-layer Multi-Layer Perceptron (`30 → 16 → 8 → 1`) with He/Xavier initialization, forward pass, binary cross-entropy loss, and full analytical matrix-calculus backpropagation.
  - **Zero Prohibited Dependencies:** 0% `torch.optim`, 0% `tensorflow.keras.optimizers`, 0% autograd engines.
- **Part A — 2D Optimizer Playground:**
  - **4 Loss Surfaces:** $L_1 (x^2+10y^2)$, $L_2 (x^2+50y^2 \text{ default})$, $L_3 (x^2+100y^2)$, and $L_4 (x^2+1000y^2)$ with condition numbers $\kappa \in [10, 1000]$.
  - **Synchronized Dual Real-Time Views:**
    - **View 1 (Contour Map):** Filled 2D contour plot, global minimum $\star (0,0)$, growing trajectories, live current position markers.
    - **View 2 (Loss Curve):** $L(\theta_t)$ vs. iteration $t$ updated in lock-step with shared color palette.
  - **Animation Engine:** Play, Pause, Step, Reset, and dynamic speed sliders.
  - **Explain-As-You-Go Panel:** Intuitive mathematical breakdowns of NAG lookahead, AdaGrad scaling, RMSProp moving average, and AdamW decoupled weight decay.
  - **Conditioning & Sensitivity Explorers:** Live sweeps over condition numbers ($\kappa$) and learning rates ($\eta \in \{0.001, 0.01, 0.1\}$) demonstrating smooth convergence vs. catastrophic divergence.
- **Part B — Real Neural Network Benchmarking:**
  - **Dataset:** Real Breast Cancer Wisconsin Diagnostic dataset (569 samples, 30 features).
  - **Dynamic UI Metrics:** Live sample counts (total, features, train, test) calculated dynamically without hardcoding.
  - **Live Training Dashboard:** Real-time epoch-by-epoch plots of training loss, validation/test loss, accuracy, and **live effective learning rate ($\eta_{\text{eff}}$) for representative weight $W_1[0,0]$**.
  - **Auto-Computed Comparison Table:** Automatically computes Final Train Loss, Final Test Loss, Train Acc, Test Acc, and **Convergence Epoch** (first epoch reaching within 1% of final validation loss).

---

## 📐 Mathematical Formulas (All Implemented from Scratch)

| Optimizer | Mathematical Update Rule | Key Characteristic |
|---|---|---|
| **1. SGD** | $\theta_{t+1} = \theta_t - \eta g_t$ | First-order steepest descent; oscillates on steep valleys |
| **2. Momentum** | $v_t = \beta v_{t-1} + (1-\beta)g_t, \quad \theta_{t+1} = \theta_t - \eta v_t$ | Velocity buffer dampens high-frequency oscillation |
| **3. NAG** | $g_{la} = \nabla L(\theta_t - \beta v_{t-1}), \quad v_t = \beta v_{t-1} + (1-\beta)g_{la}, \quad \theta_{t+1} = \theta_t - \eta v_t$ | Anticipatory lookahead braking reduces overshoot |
| **4. AdaGrad** | $G_t = G_{t-1} + g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t+\epsilon}} \odot g_t$ | Parameter-wise adaptive learning rate; slows down over time |
| **5. RMSProp** | $v_t = \beta v_{t-1} + (1-\beta)g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t+\epsilon}} \odot g_t$ | Exponential moving average fixes AdaGrad's vanishing learning rate |
| **6. Adam** | $\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \ \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \hat{m}_t$ | Combines first & second moments with early bias correction |
| **7. AdamW** | $\theta_{t+1} = \theta_t (1 - \eta \lambda) - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$ | Decoupled weight decay restores true regularization |

---

## 🏗️ Codebase Architecture

```text
Deep-Learning-Optimizer-Visualizer/
├── app.py                     # Streamlit interactive application entry point
├── src/
│   ├── __init__.py            # Package initialization & optimizer exports
│   ├── optimizers.py          # 7 pure NumPy stateful optimizers
│   ├── surfaces.py            # 2D loss surfaces (L1-L4), analytical gradients, Hessians, simulation
│   ├── neural_net.py          # BinaryMLP (Dense, ReLU, Sigmoid, BCE loss, analytical backprop)
│   ├── dataset.py             # DatasetManager (Breast cancer data, train/test split, standardization)
│   ├── visualization.py       # Plotly charts (Contour map, loss curves, training dashboard, effective LR)
│   └── experiment.py          # NNTrainingEngine, live callbacks, comparison table & convergence epoch
├── tests/
│   ├── test_optimizers.py     # Unit tests for all 7 optimizers (update rules, dict params, invalid args)
│   ├── test_surfaces.py       # Unit tests for 2D surfaces, analytical gradients, condition numbers
│   ├── test_neural_net.py     # Forward shapes, BCE loss, finite-difference numerical gradient check
│   └── test_restrictions.py   # Static analysis verifying 0 prohibited imports
├── docs/
│   └── Lab_Exercise_Optimizer_Visualizer_SGD_to_AdamW.pdf
├── requirements.txt           # Clean dependencies
├── .gitignore                 # Secure gitignore
├── REQUIREMENTS_CHECKLIST.md  # Detailed compliance matrix tracking all PDF specifications
├── REFLECTION_ANSWERS.md      # Comprehensive answers to Sections A7 (1-8) and B4 (1-16)
├── CONCLUSION.md              # 1-page academic essay on optimizer evolution + future improvements
├── DEMO_GUIDE.md              # 2-4 minute live presentation walkthrough & screenshot guide
└── README.md                  # Master documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone or Navigate to Repository
```bash
git clone https://github.com/sg721642/Deep-Learning-Optimizer-Visualizer.git
cd Deep-Learning-Optimizer-Visualizer
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Unit Tests & Verification
```bash
python3 -m pytest tests/
```

### 4. Launch Interactive Streamlit App
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 📊 Empirical Neural Network Benchmark Results (Breast Cancer Dataset)

*Auto-computed via `src/experiment.py` (80 Epochs, Batch Size 32, Seed 42):*

| Optimizer | Final Train Loss | Final Test Loss | Train Acc. | Test Acc. | Convergence Epoch (≤ 1% Final Loss) |
|---|---|---|---|---|---|
| **SGD** | 0.0657 | 0.1085 | 98.90% | 95.61% | Epoch 74 |
| **Momentum** | 0.0660 | 0.1091 | 98.68% | 95.61% | Epoch 74 |
| **NAG** | 0.0653 | 0.1085 | 98.90% | 96.49% | Epoch 76 |
| **AdaGrad** | 0.0321 | 0.0790 | 99.34% | 96.49% | Epoch 78 |
| **RMSProp** | 0.0004 | 0.4696 | 100.00% | 95.61% | Epoch 80 |
| **Adam** | 0.0062 | 0.2005 | 100.00% | 96.49% | Epoch 80 |
| **AdamW** | **0.0010** | **0.1406** | **100.00%** | **96.49%** | **Epoch 77** |

---

## 🛡️ Restrictions Compliance Audit

- [x] **Zero torch.optim or keras.optimizers:** Verified via `tests/test_restrictions.py`.
- [x] **Zero autograd engines:** Gradients derived analytically via matrix calculus.
- [x] **Unified implementation:** The identical 7 optimizer classes are reused in Part A and Part B.
- [x] **Permitted libraries only:** NumPy, Pandas, Streamlit, Plotly, Matplotlib, scikit-learn (data loader & split only).

---

## 📜 Submission Deliverables

- [x] Full source code with clean entry point `app.py`.
- [x] From-scratch implementations of all 7 optimizers in `src/optimizers.py`.
- [x] From-scratch neural network in `src/neural_net.py`.
- [x] [REFLECTION_ANSWERS.md](file:///Users/satyamgupta/Documents/Deep%20Learning%20project/REFLECTION_ANSWERS.md) answering all Section A7 & B4 questions.
- [x] [CONCLUSION.md](file:///Users/satyamgupta/Documents/Deep%20Learning%20project/CONCLUSION.md) containing 1-page optimizer evolution summary.
- [x] [DEMO_GUIDE.md](file:///Users/satyamgupta/Documents/Deep%20Learning%20project/DEMO_GUIDE.md) containing 2–4 min live demo script.
- [x] [REQUIREMENTS_CHECKLIST.md](file:///Users/satyamgupta/Documents/Deep%20Learning%20project/REQUIREMENTS_CHECKLIST.md) fully verified and audited.
