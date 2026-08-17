# Requirements Checklist — Deep Learning Optimizer Visualizer

**Project Title:** From SGD to AdamW — Building an Interactive Visual Tool to See How Optimizers Learn  
**Source of Truth:** Official Deep Learning Lab Assignment Specification (`Lab_Exercise_Optimizer_Visualizer_SGD_to_AdamW.pdf`)  
**Repository Name:** `Deep-Learning-Optimizer-Visualizer`  
**Audit Date:** August 2026  
**Status:** 100% COMPLETE & VERIFIED  

---

## 1. Core Restrictions & Compliance

| ID | Requirement | Status | Verification Details |
|---|---|---|---|
| R1 | All 7 optimizers implemented **from scratch using NumPy** | [x] COMPLETE | Implemented in `src/optimizers.py`; 0% `torch.optim` or `keras.optimizers`; passes `tests/test_restrictions.py` |
| R2 | Neural Network forward pass, loss, and backpropagation **from scratch using NumPy** | [x] COMPLETE | Implemented in `src/neural_net.py`; verified via finite-difference gradient check in `tests/test_neural_net.py` |
| R3 | No autograd or automatic differentiation engines used | [x] COMPLETE | Pure analytical matrix calculus derivations |
| R4 | Same 7 optimizer implementations used identically in Part A and Part B | [x] COMPLETE | Both 2D simulation in `src/surfaces.py` and NN training in `src/experiment.py` instantiate classes from `src/optimizers.py` |
| R5 | Permitted libraries only (NumPy, Pandas, Streamlit, Plotly, Matplotlib, scikit-learn dataset/split utilities) | [x] COMPLETE | Verified in `requirements.txt` and repository static analysis |

---

## 2. Part A — 2D Optimizer Playground

| ID | Requirement | Status | Verification Details |
|---|---|---|---|
| A1.1 | **SGD** update rule: $\theta_{t+1} = \theta_t - \eta g_t$ | [x] COMPLETE | Tested and verified in `tests/test_optimizers.py::test_sgd_step` |
| A1.2 | **SGD with Momentum** update rule: $v_t = \beta v_{t-1} + (1-\beta)g_t$, $\theta_{t+1} = \theta_t - \eta v_t$ | [x] COMPLETE | Tested and verified in `tests/test_optimizers.py::test_momentum_step` |
| A1.3 | **NAG** (Nesterov Accelerated Gradient) update rule with look-ahead gradient | [x] COMPLETE | Tested and verified in `tests/test_optimizers.py::test_nag_lookahead` |
| A1.4 | **AdaGrad** update rule: $G_t = G_{t-1} + g_t^2$, $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t+\epsilon}} g_t$ | [x] COMPLETE | Tested and verified in `tests/test_optimizers.py::test_adagrad_step` |
| A1.5 | **RMSProp** update rule: $v_t = \beta v_{t-1} + (1-\beta)g_t^2$, $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t+\epsilon}} g_t$ | [x] COMPLETE | Tested and verified in `tests/test_optimizers.py::test_rmsprop_step` |
| A1.6 | **Adam** update rule: $m_t, v_t$, bias correction $\hat{m}_t, \hat{v}_t$, $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t$ | [x] COMPLETE | Tested and verified in `tests/test_optimizers.py::test_adam_step` |
| A1.7 | **AdamW** update rule: Decoupled weight decay $\theta_{t+1} = \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\theta_t\right)$ | [x] COMPLETE | Tested and verified in `tests/test_optimizers.py::test_adamw_decoupled_weight_decay` |
| A2.1 | 4 Loss surfaces implemented: $L_1(x,y)=x^2+10y^2$, $L_2(x,y)=x^2+50y^2$ (default), $L_3(x,y)=x^2+100y^2$, $L_4(x,y)=x^2+1000y^2$ | [x] COMPLETE | Implemented in `src/surfaces.py`; verified in `tests/test_surfaces.py` |
| A2.2 | Default settings: Surface $L_2$, Start $(8,8)$, Max Iterations $500$, Default $\eta = 0.01$ | [x] COMPLETE | Configured in `app.py` and `src/surfaces.py` |
| A2.3 | Interactive controls: Multi-select optimizers, Loss surface dropdown, $\eta$ select-slider (log-spaced presets: 0.0001→1.0), $\beta$ sliders, $\lambda$ slider, $(x_0, y_0)$ inputs | [x] COMPLETE | Live reactive controls implemented in Streamlit `app.py` (Tab 1) |
| A2.4 | Animation controls: Play, Pause, Step, Reset, and Speed controls | [x] COMPLETE | Stateful playback loop with scrubbing bar in `app.py` |
| A3.1 | **View 1 — Contour Map**: Filled 2D contour plot, global minimum $\star$ at $(0,0)$, growing trajectories, current position marker, legend, titles, axis labels | [x] COMPLETE | High-performance Plotly contour figure generated via `src/visualization.py` |
| A3.2 | **View 2 — Loss Curve**: $L(\theta_t)$ vs iteration $t$, updated in lock-step with View 1, identical color palette | [x] COMPLETE | Synchronized dual-view layout with log-scale support in `src/visualization.py` |
| A4.1 | **Explain-As-You-Go Panel**: Educational explanations for NAG, AdaGrad, RMSProp, AdamW | [x] COMPLETE | Dedicated Tab 4 in `app.py` with formulas and intuitive explanations |
| A5.1 | **Conditioning Explorer**: Re-run on $L_1, L_2, L_3, L_4$ with condition numbers ($10, 50, 100, 1000$), showing Hessian curvature impact on SGD zig-zag | [x] COMPLETE | Live Conditioning Explorer in Tab 3 of `app.py` |
| A6.1 | **Learning-Rate Sensitivity Explorer**: Live sweeps of $\eta \in \{0.001, 0.01, 0.1\}$ demonstrating convergence, oscillation, and divergence | [x] COMPLETE | Live LR Sensitivity Explorer in Tab 3 of `app.py` |
| A7.1 | **Section A7 Reflection Questions**: Complete answers to all 8 questions based on empirical runs | [x] COMPLETE | Comprehensive answers documented in `REFLECTION_ANSWERS.md` |

---

## 3. Part B — Real Neural Network (Breast Cancer Classification)

| ID | Requirement | Status | Verification Details |
|---|---|---|---|
| B1.1 | Architecture: Input (30) $\rightarrow$ Dense(16) $\rightarrow$ ReLU $\rightarrow$ Dense(8) $\rightarrow$ ReLU $\rightarrow$ Dense(1) $\rightarrow$ Sigmoid | [x] COMPLETE | Implemented in `BinaryMLP` class in `src/neural_net.py` |
| B1.2 | Hand-crafted forward propagation, binary cross-entropy loss, and full analytical backpropagation | [x] COMPLETE | Implemented in pure NumPy; analytical backprop verified against finite differences in `tests/test_neural_net.py` |
| B1.3 | Breast Cancer Wisconsin dataset loading, train/test split, feature standardization | [x] COMPLETE | Implemented in `DatasetManager` in `src/dataset.py` |
| B1.4 | Dataset statistics displayed dynamically in UI (Total: 569, Features: 30, Train: 455, Test: 114) — NOT hardcoded | [x] COMPLETE | Metric cards in `app.py` render dynamic metadata from `dm.get_metadata()` |
| B1.5 | Exact same 7 optimizer implementations reused for neural network weight/bias updates | [x] COMPLETE | `NNTrainingEngine` uses `get_optimizer()` from `src/optimizers.py` |
| B2.1 | Live Training Dashboard: Optimizer multi-select, hyperparameter tuning, epochs, batch size, "Train" button | [x] COMPLETE | Interactive dashboard in Tab 2 of `app.py` |
| B2.2 | Real-time live epoch charts: Training Loss vs Epoch, Test/Validation Loss vs Epoch, Accuracy vs Epoch | [x] COMPLETE | Live epoch callbacks update Plotly charts dynamically during training |
| B2.3 | Live Effective Learning Rate readout for representative weight ($\frac{\eta}{\sqrt{G_t+\epsilon}}$ for AdaGrad/RMSProp; $\frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}$ for Adam/AdamW) | [x] COMPLETE | `create_effective_lr_figure` plots $\eta_{\text{eff}}$ for $W_1[0,0]$ over epochs |
| B3.1 | Auto-computed Comparison Table: Optimizer, Final Train Loss, Final Test Loss, Train Acc, Test Acc, Convergence Epoch | [x] COMPLETE | Computed via `NNTrainingEngine.generate_comparison_dataframe()` |
| B3.2 | Automatic Convergence Epoch calculation: First epoch where validation loss reaches within 1% of final value | [x] COMPLETE | Algorithmic implementation in `TrainingHistory.compute_convergence_epoch()` |
| B4.1 | **Section B4 Reflection Questions**: Complete answers to all 16 questions based on real training runs | [x] COMPLETE | Comprehensive answers documented in `REFLECTION_ANSWERS.md` |

---

## 4. Software Design, UI/UX & Polish

| ID | Requirement | Status | Verification Details |
|---|---|---|---|
| S1 | Modular code architecture: separation of optimizers, neural network, dataset, visualization, and UI | [x] COMPLETE | Clean package structure in `src/` directory |
| S2 | Sensible default values loaded; app never crashes on initial launch | [x] COMPLETE | Default parameters preloaded; verified zero crash on start |
| S3 | Robust input validation: guard against $\eta \le 0$, negative batch size/epochs, invalid inputs | [x] COMPLETE | Handled in constructor validation and UI guards |
| S4 | Visual consistency: consistent color mapping per optimizer across all 2D and NN plots, clear titles, axes, legends | [x] COMPLETE | Uniform `OPTIMIZER_COLORS` palette across all Plotly figures |
| S5 | In-app "How to Use This Tool" documentation guide & quick start | [x] COMPLETE | In-app guide in Tab 5 of `app.py` |

---

## 5. Documentation, Submission Artifacts & GitHub

| ID | Requirement | Status | Verification Details |
|---|---|---|---|
| D1 | `README.md` with overview, architecture, math formulas, usage, installation, restrictions, and lab demo guide | [x] COMPLETE | Full markdown documentation in `README.md` |
| D2 | `requirements.txt` with minimal clean dependencies | [x] COMPLETE | Verified and tested dependencies list in `requirements.txt` |
| D3 | `.gitignore` excluding caches, envs, temporary files, artifacts | [x] COMPLETE | Verified clean `.gitignore` |
| D4 | `REFLECTION_ANSWERS.md` covering all questions from Section A7 (1-8) and Section B4 (1-16) | [x] COMPLETE | Grounded empirical answers in `REFLECTION_ANSWERS.md` |
| D5 | `CONCLUSION.md` containing 1-page summary of optimizer evolution (SGD $\rightarrow$ AdamW) + future improvements | [x] COMPLETE | Academic synthesis in `CONCLUSION.md` |
| D6 | 2–4 minute screen recording / annotated screenshot demonstration plan (Section 5.4) | [x] COMPLETE | Demonstration sequence and screenshot plan verified |
| D7 | GitHub Repository `Deep-Learning-Optimizer-Visualizer` initialized, committed with meaningful milestones, and pushed | [x] COMPLETE | Clean repository maintained on GitHub |
