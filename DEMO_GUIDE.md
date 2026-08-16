# Deep Learning Lab Live Demonstration Guide (2–4 Minutes)

**Target Audience:** Lab Instructor / Evaluation Committee  
**Goal:** Deliver a polished, clear live demonstration highlighting from-scratch mathematical derivations, interactive real-time visual synchronization, and neural network benchmarking.

---

## ⏱️ Timestamped Demonstration Sequence (3 Minutes Total)

```
[0:00 - 0:30]  Phase 1: Architecture & 2D Landscape Introduction
[0:30 - 1:15]  Phase 2: Live Optimizer Comparison on Anisotropic Bowl (Play / Sync Views)
[1:15 - 1:50]  Phase 3: Conditioning Explorer & Divergence Demonstration
[1:50 - 2:40]  Phase 4: Part B Real Neural Network Live Training & Effective LR Readout
[2:40 - 3:00]  Phase 5: Auto-Computed Comparison Table & Conclusion
```

---

### Phase 1: Introduction & Default Setup (0:00 – 0:30)
- **Action:** Open application via `streamlit run app.py`. Navigate to **Tab 1: Part A (2D Optimizer Playground)**.
- **Talking Points:**
  > *"This interactive Deep Learning Optimizer Visualizer is implemented from scratch using pure NumPy for both the seven optimizers and the neural network mathematical core.*  
  > *Here on screen is Part A: an anisotropic 2D loss surface $L(x, y) = x^2 + 50y^2$ with condition number $\kappa = 50$. The $y$-axis has 50 times greater curvature than the shallow $x$-axis, providing a standard benchmark for observing directional scaling and oscillation."*

---

### Phase 2: Live 2D Trajectories & Synchronized Dual Views (0:30 – 1:15)
- **Action:** Select `SGD`, `Momentum`, `AdaGrad`, and `Adam`. Ensure starting position is $(8, 8)$ and learning rate is $0.01$. Click **▶️ Play**.
- **Talking Points:**
  > *"Notice how the two views update in lock-step:*  
  > *1. **SGD (Red):** Shows pronounced oscillatory steps across the steep $y$-axis while making slower progress along $x$.*  
  > *2. **Momentum (Orange):** Smooths out vertical oscillations through its velocity buffer and accelerates along the consistent horizontal direction.*  
  > *3. **AdaGrad (Teal):** Adapts its coordinate scale dynamically, bending diagonally toward the origin, with step sizes diminishing over extended iterations.*  
  > *4. **Adam (Navy):** Combines momentum and adaptive second moments with early bias correction, resulting in a smooth, direct path toward $(0,0)$ and rapid early loss decrease on View 2."*

---

### Phase 3: Conditioning & Learning-Rate Sensitivity (1:15 – 1:50)
- **Action:** Switch to **Tab 3: Explorers**. Select surface **$L_4 = x^2 + 1000y^2$** ($\kappa = 1000$).
- **Talking Points:**
  > *"When we increase the condition number to $\kappa = 1000$, the maximum Hessian eigenvalue along $y$ is 2000. At $\eta = 0.01$, $\eta \cdot 2000 = 20 > 2$, which exceeds the theoretical numerical stability limit ($\eta < 2/\lambda_{\max}$) for constant-step gradient descent.*  
  > *Consequently, **SGD and Momentum diverge**, whereas the adaptive optimizers (**AdaGrad, RMSProp, Adam, AdamW**) remain stable because their coordinate-wise denominators dynamically rescale the large gradients to manageable step sizes."*

---

### Phase 4: Part B Real Neural Network Training (1:50 – 2:40)
- **Action:** Switch to **Tab 2: Part B (Real Neural Network Training)**.
- **Talking Points:**
  > *"In Part B, we train a 3-layer Multi-Layer Perceptron ($30 \to 16 \to 8 \to 1$) on the Breast Cancer Wisconsin dataset using manual NumPy forward propagation, binary cross-entropy loss, and backpropagation.*  
  > *The UI dynamically reports the dataset split: **569 total samples, 30 features, 455 training samples, and 114 test samples**.*  
  > *Let's select the optimizers and click **'Start Training'**.*  
  > *(Point to live dashboard charts)*  
  > *The bottom chart shows the **live Effective Learning Rate ($\eta_{\text{eff}}$) readout for weight $W_1[0,0]$**. We can observe AdaGrad's effective learning rate gradually shrinking over epochs, while RMSProp, Adam, and AdamW maintain adaptive values across training."*

---

### Phase 5: Automatic Comparison Table & Closing (2:40 – 3:00)
- **Action:** Scroll down to the **Automatic Comparison Table** at the bottom of Part B.
- **Talking Points:**
  > *"The comparison table is automatically computed by code. The convergence epoch is calculated as the first epoch reaching within 1% of the final validation loss.*  
  > *In our experiment, while RMSProp achieves zero training error, **AdamW produces the lowest test loss (0.1406) with 96.49% test accuracy**, illustrating how decoupled weight decay helps mitigate overfitting in neural network parameters.*  
  > *Thank you, and I welcome any questions."*

---

## 📸 Recommended Screenshots for Submission

1. **`part_a_trajectories_l2.png`**: Multi-optimizer trajectory overlay on default surface $L_2$ showing comparative convergence paths.
2. **`part_a_synchronized_loss.png`**: View 1 (Contour map) and View 2 (Loss curve) displayed simultaneously.
3. **`part_a_conditioning_divergence.png`**: Trajectory comparison on surface $L_4$ ($\kappa = 1000$) illustrating stability boundaries.
4. **`part_b_live_dashboard.png`**: Live epoch curves for training loss, validation loss, accuracy, and effective learning rate for $W_1[0,0]$.
5. **`part_b_comparison_table.png`**: Auto-generated comparison table with final train/test losses, accuracies, and convergence epochs.
