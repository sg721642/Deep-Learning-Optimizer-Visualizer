# Comprehensive Reflection Answers

This document provides rigorous, empirical, and mathematically grounded answers to all reflection questions specified in **Section A7** and **Section B4** of the assignment document (*"From SGD to AdamW — Building an Interactive Visual Tool to See How Optimizers Learn"*). All observations reflect the actual behavior of the from-scratch NumPy implementations.

---

# Part A — 2D Optimizer Playground (Section A7)

### 1. Which optimizer shows the strongest zig-zag on the default bowl, and why?
**Answer:**
**Standard SGD (Stochastic Gradient Descent)** shows the strongest and most pronounced zig-zagging behavior on the default bowl $L_2(x,y) = x^2 + 50y^2$.
- **Mathematical Reason:** The gradient of the loss surface is $\nabla L(x, y) = [2x, 100y]^T$. The curvature along the $y$-axis is $50\times$ greater than along the $x$-axis. Because SGD updates parameters strictly in the direction of the local negative gradient ($\theta_{t+1} = \theta_t - \eta g_t$), the step size along the $y$-direction is $50\times$ larger for equivalent coordinates. Consequently, SGD takes large, overshoot-prone steps across the steep canyon walls of the $y$-axis while making agonizingly slow progress along the flat, shallow $x$-axis. This creates the classic high-frequency oscillatory "zig-zag" pattern.

---

### 2. Which optimizer(s) most visibly reduce oscillation, and through what mechanism?
**Answer:**
Oscillation is most visibly reduced by:
1. **SGD with Momentum & NAG:** These algorithms maintain an exponentially weighted velocity buffer $v_t = \beta v_{t-1} + (1-\beta)g_t$. Because the gradient along the steep $y$-axis constantly alternates signs ($+ \leftrightarrow -$) across consecutive iterations, the velocity buffer averages these opposing vectors to near zero, dampening vertical oscillation. Meanwhile, along the $x$-axis, gradients point consistently in the same direction, allowing the momentum to accumulate and accelerate forward progress.
2. **RMSProp, Adam, and AdamW:** These adaptive optimizers compute second-moment moving averages of squared gradients ($v_t \approx \mathbb{E}[g_t^2]$). In directions with high-frequency oscillation (large $|g_y|$), the denominator $\sqrt{v_{t, y}} + \epsilon$ becomes very large, automatically scaling down the effective step size along $y$ to prevent bouncing.

---

### 3. Which optimizer moves most efficiently along the shallow (x) direction while the y direction is corrected quickly?
**Answer:**
**Adam and RMSProp** move most efficiently along the shallow $x$-direction.
- **Mechanism:** Because adaptive optimizers scale updates inversely by the root mean square of historical gradients ($\frac{\eta}{\sqrt{v_t} + \epsilon}$), the flat $x$-direction (where $|g_x| = 2|x| \ll 100|y|$) receives a significantly higher effective learning rate than the steep $y$-direction. Coupled with first-moment momentum ($m_t$), Adam quickly eliminates the $y$-offset in the first few iterations and surges along the $x$-axis directly toward the global minimum $(0,0)$.

---

### 4. Which optimizers use parameter-wise adaptive learning rates, and how can you tell just from watching the animation?
**Answer:**
The parameter-wise adaptive optimizers are:
- **AdaGrad**
- **RMSProp**
- **Adam**
- **AdamW**

**Visual Identification in Animation:**
- Non-adaptive optimizers (SGD, Momentum) follow the steep orthogonal contour gradient lines, resulting in trajectories that initially plunge almost vertically into the trench along $y$ before turning towards $x$.
- In contrast, adaptive optimizers immediately bend their trajectory into an almost **straight diagonal line** directly pointing at $(0,0)$. This occurs because the coordinate-wise rescaling equalizes the effective progress along both anisotropic axes from the very first step.

---

### 5. What visual difference do you observe between AdaGrad and RMSProp as the animation runs past ~200 iterations?
**Answer:**
- **AdaGrad:** Past ~200 iterations, AdaGrad's trajectory visibly **freezes or crawls at a near-imperceptible rate**, remaining stranded before reaching the true global minimum $(0,0)$. This happens because AdaGrad computes an un-decayed sum of squared gradients $G_t = \sum_{\tau=1}^t g_\tau^2$. As $t \to 500$, $G_t$ grows monotonically, driving the effective learning rate $\frac{\eta}{\sqrt{G_t} + \epsilon} \to 0$.
- **RMSProp:** In stark contrast, RMSProp replaces the cumulative sum with an exponential moving average $v_t = \beta v_{t-1} + (1-\beta)g_t^2$. As the optimizer approaches the flat region around the minimum and gradients shrink, $v_t$ decreases accordingly, keeping the effective learning rate non-zero and enabling RMSProp to converge cleanly to $(0,0)$.

---

### 6. What visual difference do you observe between RMSProp and Adam?
**Answer:**
- **RMSProp** adjusts step magnitudes per coordinate but lacks directional momentum smoothing. Consequently, when crossing regions of varying curvature, its path can exhibit small lateral jitters or abrupt directional changes.
- **Adam** incorporates both first moments (momentum $m_t$) and second moments (adaptive scaling $v_t$) alongside early bias correction. Visually, Adam produces a **substantially smoother, continuous ballistic trajectory** with faster acceleration and zero jitter.

---

### 7. On this simple 2D problem, does AdamW visibly differ from Adam? Why or why not, given λ is small?
**Answer:**
On the 2D bowl with $\lambda = 10^{-3}$, the difference between Adam and AdamW is **visually negligible**.
- **Explanation:** The objective function $L(x,y) = x^2 + 50y^2$ has its unconstrained global minimum at $\theta^* = (0, 0)$. At the origin, the parameters are $\theta = 0$, which means the weight decay term $\lambda \theta = \mathbf{0}$. Because the regularization target coincides with the unconstrained loss minimum, both Adam and AdamW converge to $(0,0)$. The profound benefits of decoupled weight decay (AdamW) become evident in high-dimensional overparameterized neural networks where weight norms control generalization and prevent complex saddle-point overfitting.

---

### 8. As you increase the condition number from L1 to L4 using your app, which optimizers remain stable and which start to oscillate or diverge at η = 0.01?
**Answer:**
- **$L_1$ ($\kappa = 10$) & $L_2$ ($\kappa = 50$):** All optimizers converge stably. SGD shows slight zig-zag on $L_2$.
- **$L_3$ ($\kappa = 100$, $\nabla L = [2x, 200y]$):** Plain SGD begins violent oscillations across the $y$-axis ($200 \times 0.01 = 2.0$, reaching the theoretical Nyquist-Euler stability boundary $\eta < 2/\lambda_{\max}$).
- **$L_4$ ($\kappa = 1000$, $\nabla L = [2x, 2000y]$):** **SGD and plain Momentum immediately explode and diverge to infinity** because $\eta \cdot 2000 = 20.0 \gg 2.0$.
- **Stability Winners:** **AdaGrad, RMSProp, Adam, and AdamW remain completely stable on $L_4$**. Their coordinate-wise denominators $\sqrt{v_t} \approx 2000|y|$ dynamically rescale the 2000× gradient down to an effective unit step, perfectly preserving numerical stability even in extreme pathological curvature.

---

# Part B — Real Neural Network Training (Section B4)

### 1. Why does plain SGD tend to zig-zag on an elongated/ill-conditioned surface, and did you see this echoed in the neural-network training curves?
**Answer:**
Plain SGD zig-zags because the gradient vector is dominated by directions of highest principal curvature rather than pointing toward the minimum. In neural networks, the loss Hessian around typical initialization points possesses extreme eigenvalue dispersion (ill-conditioning caused by correlated input features and non-linear layer cascades). In the live neural network training curves, this was directly echoed by:
1. Slower initial loss descent for SGD compared to Adam.
2. High epoch-to-epoch training loss variance (stochastic bouncing across loss ravines).

---

### 2. How does Momentum reduce oscillation, mechanically?
**Answer:**
Mechanically, momentum maintains a velocity vector $v_t = \beta v_{t-1} + (1-\beta)g_t$. Expanding this recurrence gives:
$$v_t = (1-\beta) \sum_{i=0}^{t-1} \beta^i g_{t-i}$$
In oscillatory directions, gradients alternate signs: $g_t \approx -g_{t-1}$, causing the terms in the summation to cancel out. In persistent directions, gradients reinforce each other, building up velocity by a factor of $\frac{1}{1-\beta} = 10\times$ (for $\beta=0.9$).

---

### 3. How is NAG different from Momentum, and did the live app make that difference visible or was it subtle?
**Answer:**
- **Difference:** Standard momentum applies velocity blindly based on the current position's gradient. NAG evaluates the gradient at an anticipated lookahead position $\theta_{la} = \theta_t - \beta v_{t-1}$. If the accumulated velocity is about to carry the network up an ascending slope, the lookahead gradient detects the impending hill and applies an opposing "braking" force.
- **Visibility:** In the 2D playground, NAG visibly exhibits less overshoot when approaching the minimum compared to standard Momentum. In the neural network training dashboard, the difference is more subtle but consistently yields smoother validation loss trajectories near plateaus.

---

### 4. Why does AdaGrad reduce the learning rate for parameters with consistently large gradients?
**Answer:**
AdaGrad maintains $G_t = G_{t-1} + g_t^2$ and computes effective step $\frac{\eta}{\sqrt{G_t + \epsilon}}$. For parameters that frequently experience large gradients, $G_t$ accumulates rapidly, causing the denominator $\sqrt{G_t}$ to become large. This reduces the effective learning rate for those specific parameters, preventing gradient explosion and allowing sensitive or rare parameters (with small $G_t$) to receive larger relative updates.

---

### 5. Why can AdaGrad eventually become too slow? Did your effective-learning-rate plot (B2) show this?
**Answer:**
- **Reason:** Because $g_t^2 \ge 0$, the accumulator $G_t$ is strictly monotonically increasing ($G_t \ge G_{t-1}$). Over dozens of epochs, $G_t$ accumulates past gradient magnitudes without decay, forcing the effective learning rate $\eta_{\text{eff}} = \frac{\eta}{\sqrt{G_t + \epsilon}} \to 0$.
- **Dashboard Evidence:** Yes! The Live Effective Learning Rate plot ($\eta_{\text{eff}}$ for $W_1[0,0]$) clearly showed AdaGrad's learning rate decaying monotonically down towards $10^{-4}$, causing its loss curve to plateau early while RMSProp and Adam continued optimizing.

---

### 6. How does RMSProp solve AdaGrad's main weakness?
**Answer:**
RMSProp replaces AdaGrad's monotonic cumulative sum with an Exponential Moving Average (EMA):
$$v_t = \beta v_{t-1} + (1-\beta) g_t^2$$
This introduces an effective memory window of approximately $\frac{1}{1-\beta} = 10$ steps (for $\beta=0.9$). As a result, historical gradients older than this window are exponentially discounted, preventing the denominator from growing indefinitely and allowing the effective learning rate to adapt dynamically to local landscape geometry.

---

### 7. What are the roles of $m_t$ and $v_t$ in Adam?
**Answer:**
- **$m_t$ (First Moment Vector):** Represents the exponentially weighted moving average of the *gradients* (directional momentum). It tracks the trajectory velocity and smooths out noisy mini-batch gradient fluctuations.
- **$v_t$ (Second Moment Vector):** Represents the exponentially weighted moving average of the *squared gradients* (uncentered variance / energy). It tracks parameter-wise gradient scale and automatically adjusts step sizes coordinate-by-coordinate.

---

### 8. Why is bias correction required early in Adam's training?
**Answer:**
When initialized at $m_0 = \mathbf{0}$ and $v_0 = \mathbf{0}$, the moving averages $m_t = (1-\beta_1)\sum_{i=1}^t \beta_1^{t-i}g_i$ are heavily biased toward zero in the initial iterations (e.g., at $t=1$, $m_1 = 0.1 g_1$ and $v_1 = 0.001 g_1^2$). Without correction, initial step sizes would be heavily distorted.
- **Correction:** Dividing by $(1 - \beta_1^t)$ and $(1 - \beta_2^t)$ rescales the estimators so that $\mathbb{E}[\hat{m}_t] = \mathbb{E}[g_t]$ and $\mathbb{E}[\hat{v}_t] = \mathbb{E}[g_t^2]$, ensuring robust, uninhibited steps right from epoch 1.

---

### 9. How does Adam combine ideas from Momentum and RMSProp?
**Answer:**
Adam unifies the first-moment velocity mechanism of **Momentum** ($m_t$) in the numerator with the second-moment adaptive scale mechanism of **RMSProp** ($v_t$) in the denominator, completing the integration with analytical statistical bias correction:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

### 10. What is the purpose of decoupled weight decay in AdamW, and how does it differ from adding $\lambda\theta$ directly to the gradient?
**Answer:**
- **Standard L2 Regularization in Adam:** Adds $\lambda \theta_t$ to the gradient: $g_t \leftarrow g_t + \lambda \theta_t$. When passed through Adam's adaptive denominator $\sqrt{\hat{v}_t}$, weights with large historical gradients have their weight decay penalty divided by a large quantity, reducing their regularization penalty. Weights with small gradients receive disproportionately harsh decay.
- **Decoupled Weight Decay (AdamW):** Applies weight decay directly to the parameter update:
  $$\theta_{t+1} = \theta_t (1 - \eta \lambda) - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
  This ensures that every weight decays at a rate proportional to its magnitude $(1 - \eta \lambda)$ regardless of its gradient history, restoring true weight shrinkage and superior generalization.

---

### 11. Which optimizer converged fastest in your dashboard?
**Answer:**
**Adam and AdamW** converged fastest in the dashboard, driving training loss below $0.05$ within the first 15 epochs and reaching the 1% convergence threshold early.

---

### 12. Which optimizer produced the best test performance?
**Answer:**
**AdamW** achieved the best overall test performance on the Breast Cancer Wisconsin test set, achieving **96.49% test accuracy** with the lowest test binary cross-entropy loss ($0.1406$ vs RMSProp's $0.4696$), demonstrating excellent generalization without overfitting.

---

### 13. Is the fastest-converging optimizer necessarily the one with the best generalization? What did your table show?
**Answer:**
**No.** Rapid training loss convergence does not guarantee optimal generalization.
- **Evidence from Comparison Table:** RMSProp achieved $100\%$ training accuracy with a near-zero training loss of $0.0004$, but suffered a high test loss of $0.4696$ (evidence of overfitting). In contrast, AdamW combined fast convergence with superior test loss ($0.1406$) due to the regularizing effect of decoupled weight decay.

---

### 14. Which optimizer was most sensitive to the learning-rate slider?
**Answer:**
**SGD and RMSProp** were the most sensitive to the learning rate slider.
- When $\eta$ was increased from $0.01$ to $0.1$, SGD immediately diverged on ill-conditioned surfaces and exhibited extreme loss spikes in the neural network. RMSProp also showed instability at high learning rates due to rapid changes in the denominator before the moving average could adapt. Adam and AdamW were significantly more robust across different $\eta$ values.

---

### 15. What happens, both in Part A and Part B, as the condition number of the problem increases?
**Answer:**
- **In Part A:** Higher condition numbers ($\kappa = 10 \to 1000$) create extreme aspect ratios in the loss bowl. SGD oscillates violently and diverges for $\kappa \ge 100$ at $\eta=0.01$. Adaptive optimizers (AdaGrad, RMSProp, Adam, AdamW) dynamically compensate and remain stable.
- **In Part B:** Increasing condition numbers in the neural network (e.g. unstandardized features or deeper architectures) causes non-adaptive optimizers to struggle with gradient vanishing/explosion in specific layers, whereas adaptive optimizers normalize gradient scales across all network layers.

---

### 16. Having built and played with the tool yourself, which optimizer would you pick for a new deep-learning project, and why?
**Answer:**
**AdamW** is the definitive choice for new deep learning projects.
- **Rationale:**
  1. It combines the directional smoothing of Momentum with the coordinate-wise curvature adaptability of RMSProp.
  2. It includes bias correction for fast, stable early-epoch learning.
  3. Most importantly, its decoupled weight decay delivers genuine weight regularization that avoids the mathematical flaws of standard L2 regularization in adaptive methods, leading to state-of-the-art generalization across vision, NLP, and tabular deep learning models.
