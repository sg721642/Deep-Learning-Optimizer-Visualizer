# Lab Demonstration Guide (2–4 Minutes)

This guide outlines a concise 2–4 minute presentation workflow for demonstrating the Optimizer Visualizer during the lab evaluation.

---

## Demonstration Sequence Overview

```text
[0:00 - 0:30]  Step 1: Introduction to 2D Loss Surfaces & Setup
[0:30 - 1:15]  Step 2: Compare Optimizer Trajectories & Synchronized Views
[1:15 - 1:50]  Step 3: Conditioning & Stability Explorers
[1:50 - 2:40]  Step 4: Neural Network Training & Live Dashboard
[2:40 - 3:00]  Step 5: Comparison Table & Conclusion
```

---

### Step 1: Introduction to 2D Loss Surfaces (0:00 – 0:30)

- **Action:** Open the application (`streamlit run app.py`). Stay on **Part A — 2D Playground**.
- **What to Observe:**
  - Default surface: $L_2(x, y) = x^2 + 50y^2$ with condition number $\kappa = 50$.
  - Global minimum at $(0, 0)$ and starting position $(8, 8)$.
- **What to Explain:**
  > *"The visualizer implements seven optimizers and a 3-layer neural network from scratch using pure NumPy.*  
  > *In Part A, the default surface $L_2$ has a curvature along $y$ that is 50 times larger than along $x$, creating an anisotropic bowl to test how optimizers handle directional scaling."*

---

### Step 2: Compare Optimizer Trajectories (0:30 – 1:15)

- **Action:** Select `SGD`, `Momentum`, `AdaGrad`, and `Adam`. Set learning rate $\eta = 0.01$. Click **Play**.
- **What to Observe:**
  - **View 1 (Contour Map):** Trajectories growing point-by-point.
  - **View 2 (Loss Curve):** Synchronized loss descent across iterations.
- **What to Explain:**
  > *"As the animation plays:*  
  > *1. **SGD (Red):** Oscillates across the steep $y$-axis while making slower progress along $x$.*  
  > *2. **Momentum (Orange):** Dampens vertical oscillations and accelerates along $x$ using its velocity buffer.*  
  > *3. **AdaGrad (Teal):** Adapts coordinate scales independently, bending diagonally toward the origin, though step sizes diminish over longer runs.*  
  > *4. **Adam (Navy):** Combines momentum and adaptive second moments with bias correction, producing a smooth trajectory directly toward the minimum."*

---

### Step 3: Conditioning & Stability Explorers (1:15 – 1:50)

- **Action:** Switch to the **Conditioning & LR Sensitivity** tab. Select surface $L_4(x, y) = x^2 + 1000y^2$ ($\kappa = 1000$).
- **What to Observe:**
  - Trajectory comparison across $L_1$ through $L_4$.
  - Sensitivity curves for $\eta \in \{0.001, 0.01, 0.1\}$.
- **What to Explain:**
  > *"On $L_4$, the condition number is 1000. At $\eta = 0.01$, constant-step gradient descent exceeds the stability bound ($\eta > 2/\lambda_{\max} = 0.001$), causing SGD to diverge.*  
  > *In contrast, adaptive optimizers scale updates inversely by recent gradient magnitudes, maintaining stable convergence."*

---

### Step 4: Neural Network Training & Live Dashboard (1:50 – 2:40)

- **Action:** Switch to **Part B — Neural Network**. Review the dynamic dataset statistics (569 samples, 30 features, 455 train, 114 test). Select all optimizers and click **Start Training**.
- **What to Observe:**
  - Live epoch updates of training loss, validation loss, and accuracy.
  - Live **Effective Learning Rate ($\eta_{\text{eff}}$)** plot for representative weight $W_1[0,0]$.
- **What to Explain:**
  > *"Part B trains an MLP ($30 \to 16 \to 8 \to 1$) on the Breast Cancer Wisconsin dataset using manual backpropagation.*  
  > *The effective learning rate plot empirically shows AdaGrad's learning rate monotonically shrinking over epochs, while RMSProp and Adam maintain adaptive step scales."*

---

### Step 5: Comparison Table & Summary (2:40 – 3:00)

- **Action:** Scroll to the **Comparison Table** generated at the end of training.
- **What to Observe:**
  - Final train loss, test loss, train accuracy, test accuracy, and auto-computed convergence epoch (first epoch reaching within 1% of final validation loss).
- **What to Explain:**
  > *"The comparison table summarizes training metrics across all selected optimizers.*  
  > *AdamW applies decoupled weight decay directly to parameters, achieving lower test loss on this benchmark by preventing the regularization distortion present in standard adaptive gradient methods."*

---

## Suggested Screenshots for Submission

1. **`part_a_trajectories_l2.png`**: Multi-optimizer trajectory overlay on surface $L_2$.
2. **`part_a_synchronized_views.png`**: Dual synchronized views (Contour map and Loss curve).
3. **`part_a_conditioning_explorer.png`**: Trajectory comparison across $L_1$ to $L_4$.
4. **`part_b_live_dashboard.png`**: Live loss, accuracy, and effective learning rate curves.
5. **`part_b_comparison_table.png`**: Auto-computed summary table with convergence epochs.
