# Deep Learning Lab Live Demonstration Guide (2–4 Minutes)

**Target Audience:** Lab Instructor / Evaluation Committee  
**Goal:** Deliver a polished, confident live demonstration highlighting from-scratch mathematical derivations, interactive real-time visual synchronization, and neural network benchmarking for maximum marks.

---

## ⏱️ Precise Timestamped Demo Script (3 Minutes Total)

```
[0:00 - 0:30]  Phase 1: Architecture & 2D Landscape Introduction
[0:30 - 1:15]  Phase 2: Live Optimizer Comparison on Anisotropic Bowl (Play / Sync Views)
[1:15 - 1:50]  Phase 3: Conditioning Explorer & Divergence Demonstration
[1:50 - 2:40]  Phase 4: Part B Real Neural Network Live Training & Effective LR Readout
[2:40 - 3:00]  Phase 5: Auto-Computed Comparison Table & Conclusion
```

---

### Phase 1: Introduction & Default Setup (0:00 – 0:30)
- **Action:** Open application via `streamlit run app.py`. Stay on **Tab 1: Part A (2D Optimizer Playground)**.
- **Talking Points:**
  > *"Good morning. Today I am demonstrating our interactive Deep Learning Optimizer Visualizer, built completely from scratch using pure NumPy for both the seven optimizers and the neural network mathematical core.*  
  > *Here on screen is Part A: an anisotropic 2D loss surface $L(x, y) = x^2 + 50y^2$ with condition number $\kappa = 50$. The steep $y$-axis has 50 times greater curvature than the shallow $x$-axis, making it the classic benchmark for optimizer oscillation."*

---

### Phase 2: Live 2D Trajectories & Synchronized Dual Views (0:30 – 1:15)
- **Action:** Select `SGD`, `Momentum`, `AdaGrad`, and `Adam`. Ensure starting position is $(8, 8)$ and learning rate is $0.01$. Click **▶️ Play**.
- **Talking Points:**
  > *"Notice how the two views update in real-time lock-step:*  
  > *1. **SGD (Red):** Takes violent orthogonal jumps across the $y$-axis, bouncing repeatedly with severe zig-zagging.*  
  > *2. **Momentum (Orange):** Smooths out vertical oscillations through its velocity buffer and surges forward.*  
  > *3. **AdaGrad (Teal):** Rapidly adapts its coordinate scale, bending diagonally toward the origin, but slows down as iterations exceed 200.*  
  > *4. **Adam (Navy):** Combines momentum and adaptive second moments with bias correction to carve a direct ballistic line straight into the global minimum $\star (0,0)$. On View 2, Adam’s loss drops exponentially faster than all other methods."*

---

### Phase 3: Conditioning & Learning-Rate Sensitivity (1:15 – 1:50)
- **Action:** Switch to **Tab 3: Explorers**. Select surface **$L_4 = x^2 + 1000y^2$** ($\kappa = 1000$).
- **Talking Points:**
  > *"When we increase the condition number to $\kappa = 1000$, the Hessian eigenvalue along $y$ becomes 2000. At $\eta = 0.01$, $\eta \cdot 2000 = 20 > 2$, which violates the mathematical stability bound for standard gradient descent.*  
  > *As you can see, **SGD and Momentum diverge instantly to infinity**, whereas our adaptive optimizers (**AdaGrad, RMSProp, Adam, AdamW**) remain perfectly stable because their denominators dynamically normalize the 2000× gradient down to unit scale."*

---

### Phase 4: Part B Real Neural Network Training (1:50 – 2:40)
- **Action:** Switch to **Tab 2: Part B (Real Neural Network Training)**.
- **Talking Points:**
  > *"In Part B, we extend the tool to train a 3-layer Multi-Layer Perceptron ($30 \to 16 \to 8 \to 1$) on the Breast Cancer Wisconsin dataset using manual NumPy forward, binary cross-entropy, and backpropagation.*  
  > *Our UI dynamically reports dataset statistics: **569 total samples, 30 features, 455 training samples, and 114 test samples**.*  
  > *Let's select all optimizers and click **'Start Training'**.*  
  > *(Point to the live dashboard)*  
  > *Notice the bottom chart: this is the **live Effective Learning Rate ($\eta_{\text{eff}}$) readout for weight $W_1[0,0]$**. We can empirically witness AdaGrad's learning rate decaying monotonically to near zero, while RMSProp, Adam, and AdamW maintain healthy adaptive ranges."*

---

### Phase 5: Automatic Comparison Table & Closing (2:40 – 3:00)
- **Action:** Scroll down to the **Automatic Comparison Table** generated at the bottom of Part B.
- **Talking Points:**
  > *"The comparison table is automatically computed by code. The convergence epoch is strictly calculated as the first epoch reaching within 1% of the final validation loss.*  
  > *Notice that while RMSProp achieves 100% training accuracy with 0.0004 loss, **AdamW achieves the best test loss (0.1406) and 96.49% test accuracy**, demonstrating how decoupled weight decay prevents overfitting.*  
  > *Thank you, and I am happy to answer any questions!"*

---

## 📸 Recommended Screenshots for Submission

1. **`part_a_trajectories_l2.png`**: Multi-optimizer trajectory overlay on default surface $L_2$ showing SGD oscillation vs Adam direct path.
2. **`part_a_synchronized_loss.png`**: View 1 and View 2 synchronized side-by-side.
3. **`part_a_conditioning_divergence.png`**: Divergence on surface $L_4$ ($\kappa = 1000$) demonstrating stability limits.
4. **`part_b_live_dashboard.png`**: Live epoch curves for training/test loss, accuracy, and effective learning rate for $W_1[0,0]$.
5. **`part_b_comparison_table.png`**: Auto-generated comparison table with final train/test losses, accuracies, and convergence epochs.
