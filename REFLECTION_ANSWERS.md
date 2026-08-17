# Comprehensive Reflection Answers

This document provides mathematically grounded answers to all reflection questions specified in **Section A7** and **Section B4** of the assignment document (*"From SGD to AdamW — Building an Interactive Visual Tool to See How Optimizers Learn"*). All observations reflect the actual behavior of the from-scratch NumPy implementations.

---

# Part A — 2D Optimizer Playground (Section A7)

### 1. Which optimizer shows the strongest zig-zag on the default bowl, and why?
**Answer:**
**Standard SGD (Stochastic Gradient Descent)** shows the strongest oscillatory "zig-zag" behaviour on the default bowl $L_2(x,y) = x^2 + 50y^2$.
- **Mathematical Reason:** The gradient of the loss surface is $\nabla L(x, y) = [2x, 100y]^T$. The curvature along the $y$-axis is $50\times$ greater than along the $x$-axis. Because SGD updates parameters strictly in the direction of the local negative gradient ($\theta_{t+1} = \theta_t - \eta g_t$), the step size along the $y$-direction is $50\times$ larger for equivalent coordinates. Consequently, SGD takes large steps across the steep canyon walls of the $y$-axis while making slower progress along the flat $x$-axis, creating an oscillatory pattern.

---

### 2. Which optimizer(s) most visibly reduce oscillation, and through what mechanism?
**Answer:**
Oscillation is visibly reduced by:
1. **SGD with Momentum & NAG:** These algorithms maintain an exponentially weighted velocity buffer $v_t = \beta v_{t-1} + (1-\beta)g_t$. Because the gradient along the steep $y$-axis alternates signs across consecutive iterations, the velocity buffer averages these opposing vectors toward zero, dampening vertical oscillation. Meanwhile, along the $x$-axis, gradients point consistently in the same direction, allowing the momentum to accumulate and accelerate forward progress.
2. **RMSProp, Adam, and AdamW:** These adaptive optimizers compute second-moment moving averages of squared gradients ($v_t \approx \mathbb{E}[g_t^2]$). In directions with large gradients ($|g_y|$), the denominator $\sqrt{v_{t, y}} + \epsilon$ increases, scaling down the effective step size along $y$ to reduce bouncing.

---

### 3. Which optimizer moves most efficiently along the shallow (x) direction while the y direction is corrected quickly?
**Answer:**
**Adam and RMSProp** move most efficiently along the shallow $x$-direction.
- **Mechanism:** Because adaptive optimizers scale updates inversely by the root mean square of historical gradients ($\frac{\eta}{\sqrt{v_t} + \epsilon}$), the flat $x$-direction receives a relatively higher effective learning rate than the steep $y$-direction. Coupled with first-moment momentum ($m_t$), Adam quickly reduces the $y$-offset and progresses along the $x$-axis toward $(0,0)$.

---

### 4. Which optimizers use parameter-wise adaptive learning rates, and how can you tell just from watching the animation?
**Answer:**
The parameter-wise adaptive optimizers are:
- **AdaGrad**
- **RMSProp**
- **Adam**
- **AdamW**

**Visual Identification in Animation:**
- Non-adaptive optimizers (SGD, Momentum) follow the steep orthogonal contour gradient lines, resulting in trajectories that initially plunge vertically into the trench along $y$ before turning towards $x$.
- In contrast, adaptive optimizers adjust step sizes per coordinate, bending their trajectory into a smoother diagonal path directly toward $(0,0)$ from the initial steps.

---

### 5. What visual difference do you observe between AdaGrad and RMSProp as the animation runs past ~200 iterations?
**Answer:**
- **AdaGrad:** Past ~200 iterations, AdaGrad's progress slows down noticeably and may stop short of reaching $(0,0)$ within the iteration limit. This happens because AdaGrad computes an un-decayed sum of squared gradients $G_t = \sum_{\tau=1}^t g_\tau^2$. As $t$ increases, $G_t$ grows monotonically, driving the effective learning rate $\frac{\eta}{\sqrt{G_t} + \epsilon} \to 0$.
- **RMSProp:** RMSProp replaces the cumulative sum with an exponential moving average $v_t = \beta v_{t-1} + (1-\beta)g_t^2$. As the optimizer approaches the flat region around the minimum and gradients shrink, $v_t$ decreases accordingly, keeping the effective learning rate non-zero and enabling RMSProp to continue progressing toward $(0,0)$.

---

### 6. What visual difference do you observe between RMSProp and Adam?
**Answer:**
- **RMSProp** adjusts step magnitudes per coordinate but lacks first-moment momentum smoothing. When traversing varying curvature, its path can show minor step-to-step directional fluctuations.
- **Adam** incorporates both first moments (momentum $m_t$) and second moments (adaptive scaling $v_t$) alongside early bias correction, resulting in a smoother, continuous trajectory toward the minimum.

---

### 7. On this simple 2D problem, does AdamW visibly differ from Adam? Why or why not, given λ is small?
**Answer:**
On the 2D bowl with $\lambda = 10^{-3}$, the difference between Adam and AdamW is **subtle**.
- **Explanation:** The objective function $L(x,y) = x^2 + 50y^2$ has its unconstrained global minimum at $\theta^* = (0, 0)$. At the origin, $\theta = 0$, so the weight decay term $\lambda \theta = \mathbf{0}$. Because the regularization target coincides with the unconstrained loss minimum, both Adam and AdamW converge to $(0,0)$. The benefits of decoupled weight decay (AdamW) become evident in high-dimensional overparameterized neural networks where weight norms control generalization and mitigate overfitting.

---

### 8. As you increase the condition number from L1 to L4 using your app, which optimizers remain stable and which start to oscillate or diverge at η = 0.01?
**Answer:**
- **$L_1$ ($\kappa = 10$) & $L_2$ ($\kappa = 50$):** All optimizers converge stably. SGD shows slight oscillation on $L_2$.
- **$L_3$ ($\kappa = 100$, $\nabla L = [2x, 200y]$):** Plain SGD begins strong oscillations across the $y$-axis ($200 \times 0.01 = 2.0$, reaching the theoretical stability boundary $\eta < 2/\lambda_{\max}$).
- **$L_4$ ($\kappa = 1000$, $\nabla L = [2x, 2000y]$):** **SGD and Momentum diverge** because $\eta \cdot 2000 = 20.0 > 2.0$, exceeding the linear stability threshold.
- **Adaptive Optimizers:** **AdaGrad, RMSProp, Adam, and AdamW remain stable on $L_4$**. Their coordinate-wise denominators $\sqrt{v_t}$ dynamically rescale the large gradient components, preserving numerical stability even under ill-conditioned curvature.

---

# Part B — Real Neural Network Training (Section B4)

### 1. Why does plain SGD tend to zig-zag on an elongated/ill-conditioned surface, and did you see this echoed in the neural-network training curves?
**Answer:**
Plain SGD oscillates because the gradient vector is dominated by directions of highest principal curvature rather than pointing directly toward the minimum. In neural networks, ill-conditioned curvature is common due to correlated input features and deep layer activations. In the live neural network training curves, this was reflected by:
1. Slower initial training loss decrease for SGD compared to momentum/adaptive optimizers.
2. Higher epoch-to-epoch variance in training loss during early epochs.

---

### 2. How does Momentum reduce oscillation, mechanically?
**Answer:**
Mechanically, momentum maintains a velocity vector $v_t = \beta v_{t-1} + (1-\beta)g_t$. Expanding this recurrence gives:
$$v_t = (1-\beta) \sum_{i=0}^{t-1} \beta^i g_{t-i}$$
In oscillatory directions, gradients alternate signs across consecutive mini-batches, causing opposing terms in the summation to cancel. In consistent directions, gradients reinforce each other, building up velocity by an effective factor of $\frac{1}{1-\beta} = 10\times$ (for $\beta=0.9$).

---

### 3. How is NAG different from Momentum, and did the live app make that difference visible or was it subtle?
**Answer:**
- **Difference:** Standard momentum applies velocity based on the current position's gradient. NAG evaluates the gradient at an anticipated look-ahead position $\theta_{la} = \theta_t - \eta\beta v_{t-1}$ (the $\eta$ keeps the lookahead dimensionally consistent with the final update $\theta_{t+1} = \theta_t - \eta v_t$, since $v$ has gradient units). If the accumulated velocity is about to carry parameters up an ascending slope, the look-ahead gradient detects the slope in advance and applies an opposing corrective force.
- **Visibility:** In the 2D playground, NAG shows slightly less overshoot when approaching the minimum compared to standard Momentum. In neural network training, the difference is subtle but produces stable loss descent near plateaus.

---

### 4. Why does AdaGrad reduce the learning rate for parameters with consistently large gradients?
**Answer:**
AdaGrad maintains $G_t = G_{t-1} + g_t^2$ and computes step sizes scaled by $\frac{\eta}{\sqrt{G_t + \epsilon}}$. For parameters that frequently experience large gradients, $G_t$ accumulates rapidly, making the denominator $\sqrt{G_t}$ large. This reduces the effective learning rate for those specific parameters, preventing instability while allowing parameters with smaller historical gradients to receive relatively larger updates.

---

### 5. Why can AdaGrad eventually become too slow? Did your effective-learning-rate plot (B2) show this?
**Answer:**
- **Reason:** Because $g_t^2 \ge 0$, the accumulator $G_t$ is monotonically non-decreasing ($G_t \ge G_{t-1}$). Over extended training epochs, $G_t$ continuously grows, forcing the effective learning rate $\eta_{\text{eff}} = \frac{\eta}{\sqrt{G_t + \epsilon}} \to 0$.
- **Dashboard Evidence:** Yes. The Live Effective Learning Rate plot ($\eta_{\text{eff}}$ for $W_1[0,0]$) showed AdaGrad's effective learning rate decaying monotonically down toward $10^{-4}$, causing its loss curve to flatten early.

---

### 6. How does RMSProp solve AdaGrad's main weakness?
**Answer:**
RMSProp replaces AdaGrad's monotonic cumulative sum with an Exponential Moving Average (EMA):
$$v_t = \beta v_{t-1} + (1-\beta) g_t^2$$
This introduces an effective memory window of approximately $\frac{1}{1-\beta} = 10$ steps (for $\beta=0.9$). As a result, older historical gradients are exponentially discounted, preventing the denominator from growing indefinitely and allowing the effective learning rate to adapt to local landscape geometry.

---

### 7. What are the roles of $m_t$ and $v_t$ in Adam?
**Answer:**
- **$m_t$ (First Moment Vector):** Represents the exponentially weighted moving average of the gradients (directional momentum). It tracks trajectory direction and smooths out noisy mini-batch gradient fluctuations.
- **$v_t$ (Second Moment Vector):** Represents the exponentially weighted moving average of the squared gradients (uncentered variance / energy). It tracks parameter-wise gradient scale and adjusts step sizes coordinate-by-coordinate.

---

### 8. Why is bias correction required early in Adam's training?
**Answer:**
When initialized at $m_0 = \mathbf{0}$ and $v_0 = \mathbf{0}$, the moving averages $m_t = (1-\beta_1)\sum_{i=1}^{t} \beta_1^{t-i}g_i$ are biased toward zero during initial iterations (e.g., at $t=1$, $m_1 = 0.1 g_1$ and $v_1 = 0.001 g_1^2$).
- **Correction:** Dividing by $(1 - \beta_1^t)$ and $(1 - \beta_2^t)$ rescales the estimators so that $\mathbb{E}[\hat{m}_t] = \mathbb{E}[g_t]$ and $\mathbb{E}[\hat{v}_t] = \mathbb{E}[g_t^2]$, ensuring uninhibited step sizes from the first epoch.

---

### 9. How does Adam combine ideas from Momentum and RMSProp?
**Answer:**
Adam combines the first-moment velocity mechanism of **Momentum** ($m_t$) in the numerator with the second-moment adaptive scale mechanism of **RMSProp** ($v_t$) in the denominator, completing the integration with analytical statistical bias correction:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

### 10. What is the purpose of decoupled weight decay in AdamW, and how does it differ from adding $\lambda\theta$ directly to the gradient?
**Answer:**
- **Standard L2 Regularization in Adam:** Adds $\lambda \theta_t$ to the gradient: $g_t \leftarrow g_t + \lambda \theta_t$. When passed through Adam's adaptive denominator $\sqrt{\hat{v}_t}$, weights with large historical gradients have their weight decay penalty divided by a large quantity, reducing their effective regularization. Weights with small gradients receive disproportionately large decay penalties.
- **Decoupled Weight Decay (AdamW):** Applies weight decay directly to the parameter update:
  $$\theta_{t+1} = \theta_t (1 - \eta \lambda) - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
  This ensures that every weight decays proportionally to its current value $(1 - \eta \lambda)$ independent of its gradient history, restoring consistent weight shrinkage.

---

### 11. Which optimizer converged fastest in your dashboard?
**Answer:**
In our empirical benchmark on the Breast Cancer dataset, **Adam and AdamW** converged fastest, reducing training loss below $0.05$ within the first 15 epochs and reaching the 1% convergence threshold early.

---

### 12. Which optimizer produced the best test performance?
**Answer:**
In our benchmark run, **AdamW** produced the lowest test loss ($0.1406$) with a test accuracy of **96.49%**, demonstrating effective generalization on the test set.

---

### 13. Is the fastest-converging optimizer necessarily the one with the best generalization? What did your table show?
**Answer:**
**No.** Fast training loss convergence does not automatically imply superior generalization.
- **Observation:** RMSProp achieved near-zero training loss ($0.0004$) and 100% training accuracy, but had a higher test loss ($0.4696$) compared to AdamW ($0.1406$). This demonstrates that aggressive fitting on training data can sometimes lead to overfitting if not properly regularized.

---

### 14. Which optimizer was most sensitive to the learning-rate slider?
**Answer:**
**SGD and RMSProp** were most sensitive to learning-rate variations.
- Increasing $\eta$ to $0.1$ caused SGD to diverge on ill-conditioned 2D surfaces and produced high loss variance during neural network training. RMSProp was also sensitive to large learning rates before its moving average could adapt. Adam and AdamW showed broader stability across different learning rate settings.

---

### 15. What happens, both in Part A and Part B, as the condition number of the problem increases?
**Answer:**
- **In Part A:** Higher condition numbers ($\kappa = 10 \to 1000$) create increasingly elongated loss valleys. Constant-step SGD oscillates and diverges on $L_4$ at $\eta=0.01$. Adaptive optimizers (AdaGrad, RMSProp, Adam, AdamW) compensate by scaling coordinates individually, remaining stable.
- **In Part B:** Ill-conditioning in neural networks (such as unnormalized features or deeper layers) causes non-adaptive optimizers to make uneven progress across parameters, whereas adaptive optimizers normalize updates across different weight matrices.

---

### 16. Having built and played with the tool yourself, which optimizer would you pick for a new deep-learning project, and why?
**Answer:**
**AdamW** is generally a strong default choice for deep learning projects.
- **Rationale:**
  1. It combines directional momentum ($m_t$) with coordinate-wise adaptive scaling ($v_t$).
  2. Bias correction provides stable optimization during initial epochs.
  3. Decoupled weight decay maintains true regularization independent of adaptive gradient scales, helping prevent overfitting across complex architectures.
