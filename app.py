"""
From SGD to AdamW — Building an Interactive Visual Tool to See How Optimizers Learn
Interactive Streamlit Application
"""
from typing import Dict, List, Any, Optional, Tuple
import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.optimizers import (
    OPTIMIZER_COLORS,
    OPTIMIZER_DESCRIPTIONS,
    get_optimizer,
    SGD,
    Momentum,
    NAG,
    AdaGrad,
    RMSProp,
    Adam,
    AdamW
)
from src.surfaces import SURFACES, DEFAULT_SURFACE_KEY, get_surface
from src.dataset import DatasetManager
from src.neural_net import BinaryMLP
from src.experiment import NNTrainingEngine, TrainingHistory
from src.visualization import (
    create_contour_figure,
    create_loss_curve_figure,
    create_nn_loss_figure,
    create_nn_accuracy_figure,
    create_effective_lr_figure
)

# Page configuration
st.set_page_config(
    page_title="Deep Learning Optimizer Visualizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look & typography
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        color: #F8F9FA;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #ADB5BD;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #1E222A;
        border: 1px solid #2D333B;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 600;
        color: #58A6FF;
    }
    .info-box {
        background-color: #161B22;
        border-left: 4px solid #58A6FF;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "anim_step" not in st.session_state:
    st.session_state.anim_step = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "dataset_manager" not in st.session_state:
    st.session_state.dataset_manager = DatasetManager(test_size=0.2, random_state=42)
if "nn_histories" not in st.session_state:
    st.session_state.nn_histories = {}
if "nn_is_training" not in st.session_state:
    st.session_state.nn_is_training = False


# Title & Header
st.markdown("<div class='main-title'>⚡ From SGD to AdamW — Optimizer Visualizer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>An Interactive Visual Tool to See How Optimizers Learn from First Principles (Pure NumPy)</div>", unsafe_allow_html=True)


# Main Tabs Navigation
tabs = st.tabs([
    "🎯 Part A: 2D Optimizer Playground",
    "🧠 Part B: Real Neural Network Training",
    "🔬 Conditioning & LR Sensitivity Explorers",
    "💡 Explain-As-You-Go",
    "📖 How to Use This Tool"
])


# ==============================================================================
# TAB 1: PART A — 2D OPTIMIZER PLAYGROUND
# ==============================================================================
with tabs[0]:
    col_ctrl, col_viz = st.columns([1, 2.5], gap="medium")

    with col_ctrl:
        st.markdown("### 🎛️ Playground Controls")
        
        # 1. Loss surface selector
        surface_key = st.selectbox(
            "Select 2D Loss Surface",
            options=list(SURFACES.keys()),
            index=list(SURFACES.keys()).index(DEFAULT_SURFACE_KEY),
            help="Choose between L1 (mild bowl) to L4 (ill-conditioned narrow valley)",
            key="part_a_surface_select"
        )
        selected_surface = get_surface(surface_key)

        st.info(f"**Condition Number (κ):** {selected_surface.condition_number:.0f}  \n"
                f"**Formula:** $L(x, y) = {selected_surface.formula_str}$  \n"
                f"**Global Min:** (0, 0)")

        # 2. Multi-select optimizers
        all_opts = ["SGD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam", "AdamW"]
        selected_opts = st.multiselect(
            "Select Optimizers to Overlay",
            options=all_opts,
            default=["SGD", "Momentum", "AdaGrad", "Adam"],
            help="Select one or more optimizers to compare trajectories simultaneously",
            key="part_a_optimizer_multiselect"
        )

        if not selected_opts:
            st.warning("Please select at least one optimizer.")
            selected_opts = ["Adam"]

        st.markdown("#### ⚙️ Hyperparameters")
        
        # Learning rate
        lr = st.number_input(
            "Learning Rate (η)",
            min_value=0.0001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.4f",
            help="Default is 0.01. Controls step magnitude along the gradient.",
            key="part_a_lr_input"
        )

        # Optimizer-specific parameters inside expander
        with st.expander("Advanced Optimizer Parameters", expanded=False):
            beta_momentum = st.slider("Momentum / NAG β", min_value=0.0, max_value=0.99, value=0.9, step=0.05, key="part_a_beta_momentum")
            beta_rmsprop = st.slider("RMSProp β", min_value=0.0, max_value=0.99, value=0.9, step=0.05, key="part_a_beta_rmsprop")
            beta1_adam = st.slider("Adam/AdamW β₁", min_value=0.0, max_value=0.99, value=0.9, step=0.05, key="part_a_beta1_adam")
            beta2_adam = st.slider("Adam/AdamW β₂", min_value=0.5, max_value=0.9999, value=0.999, step=0.001, format="%.4f", key="part_a_beta2_adam")
            weight_decay = st.number_input("AdamW Weight Decay (λ)", min_value=0.0, max_value=0.5, value=0.001, step=0.001, format="%.4f", key="part_a_lambda_adamw")

        # Starting point
        st.markdown("#### 📍 Starting Position & Steps")
        col_x0, col_y0 = st.columns(2)
        with col_x0:
            x0 = st.number_input("Initial x₀", value=8.0, step=0.5, key="part_a_x0_input")
        with col_y0:
            y0 = st.number_input("Initial y₀", value=8.0, step=0.5, key="part_a_y0_input")

        max_steps = st.slider("Max Iterations", min_value=50, max_value=500, value=300, step=25, key="part_a_max_steps_slider")

        # Simulation Computation
        trajectories: Dict[str, np.ndarray] = {}
        losses_dict: Dict[str, np.ndarray] = {}

        for opt_name in selected_opts:
            if opt_name == "SGD":
                opt = SGD(lr=lr)
            elif opt_name == "Momentum":
                opt = Momentum(lr=lr, beta=beta_momentum)
            elif opt_name == "NAG":
                opt = NAG(lr=lr, beta=beta_momentum)
            elif opt_name == "AdaGrad":
                opt = AdaGrad(lr=lr)
            elif opt_name == "RMSProp":
                opt = RMSProp(lr=lr, beta=beta_rmsprop)
            elif opt_name == "Adam":
                opt = Adam(lr=lr, beta1=beta1_adam, beta2=beta2_adam)
            elif opt_name == "AdamW":
                opt = AdamW(lr=lr, beta1=beta1_adam, beta2=beta2_adam, weight_decay=weight_decay)
            else:
                opt = SGD(lr=lr)

            traj, loss_hist = selected_surface.simulate_trajectory(
                optimizer=opt,
                start_point=(x0, y0),
                max_iters=max_steps
            )
            trajectories[opt_name] = traj
            losses_dict[opt_name] = loss_hist

        # Determine effective maximum trajectory length
        actual_max_len = max([len(t) for t in trajectories.values()]) if trajectories else max_steps

        # Animation Controls
        st.markdown("#### 🎬 Animation Controls")
        c_play, c_pause, c_step, c_reset = st.columns(4)
        
        with c_play:
            if st.button("▶️ Play", use_container_width=True, key="btn_play"):
                st.session_state.is_playing = True
        with c_pause:
            if st.button("⏸️ Pause", use_container_width=True, key="btn_pause"):
                st.session_state.is_playing = False
        with c_step:
            if st.button("⏭️ Step", use_container_width=True, key="btn_step"):
                st.session_state.is_playing = False
                st.session_state.anim_step = min(st.session_state.anim_step + 5, actual_max_len - 1)
        with c_reset:
            if st.button("🔄 Reset", use_container_width=True, key="btn_reset"):
                st.session_state.is_playing = False
                st.session_state.anim_step = 0

        anim_speed = st.select_slider(
            "Animation Speed",
            options=["Slow (1x)", "Normal (2x)", "Fast (5x)", "Instant (Max)"],
            value="Fast (5x)",
            key="anim_speed_slider"
        )
        step_increment = {"Slow (1x)": 1, "Normal (2x)": 3, "Fast (5x)": 8, "Instant (Max)": 25}[anim_speed]

        # Slider for manual scrubbing
        st.session_state.anim_step = st.slider(
            "Iteration Scrub Bar",
            min_value=0,
            max_value=max(actual_max_len - 1, 1),
            value=min(st.session_state.anim_step, max(actual_max_len - 1, 1)),
            step=1,
            key="part_a_scrub_bar"
        )

    with col_viz:
        st.markdown("### 📊 Synchronized Real-Time Views")
        v1_col, v2_col = st.columns(2, gap="small")
        
        with v1_col:
            fig_contour = create_contour_figure(
                surface=selected_surface,
                trajectories=trajectories,
                current_step=st.session_state.anim_step,
                start_point=(x0, y0),
                x_range=(-max(abs(x0) + 2, 10), max(abs(x0) + 2, 10)),
                y_range=(-max(abs(y0) + 2, 10), max(abs(y0) + 2, 10))
            )
            st.plotly_chart(fig_contour, use_container_width=True, key="fig_contour_chart")

        with v2_col:
            fig_loss = create_loss_curve_figure(
                losses_dict=losses_dict,
                current_step=st.session_state.anim_step,
                log_scale=True
            )
            st.plotly_chart(fig_loss, use_container_width=True, key="fig_loss_chart")

        # Trajectory Step Summary Metrics
        st.markdown("##### 📈 Current Step Optimizer Status")
        m_cols = st.columns(len(selected_opts))
        for idx, opt_name in enumerate(selected_opts):
            with m_cols[idx]:
                traj = trajectories[opt_name]
                step_idx = min(st.session_state.anim_step, len(traj) - 1)
                curr_pos = traj[step_idx]
                curr_loss = losses_dict[opt_name][step_idx]
                st.markdown(f"""
                <div style="border-left: 3px solid {OPTIMIZER_COLORS[opt_name]}; padding-left: 8px; margin-bottom: 8px;">
                    <strong>{opt_name}</strong><br>
                    <small>Pos: ({curr_pos[0]:.2f}, {curr_pos[1]:.2f})</small><br>
                    <small>Loss: <b>{curr_loss:.4e}</b></small>
                </div>
                """, unsafe_allow_html=True)

    # Automatic animation loop step
    if st.session_state.is_playing and st.session_state.anim_step < actual_max_len - 1:
        st.session_state.anim_step = min(st.session_state.anim_step + step_increment, actual_max_len - 1)
        time.sleep(0.05)
        st.rerun()
    elif st.session_state.is_playing and st.session_state.anim_step >= actual_max_len - 1:
        st.session_state.is_playing = False


# ==============================================================================
# TAB 2: PART B — REAL NEURAL NETWORK TRAINING
# ==============================================================================
with tabs[1]:
    st.markdown("### 🧠 Multi-Layer Perceptron on Breast Cancer Wisconsin Dataset")
    st.markdown("Train a 3-layer neural network from scratch using pure NumPy backpropagation and evaluate the 7 custom optimizers.")

    dm: DatasetManager = st.session_state.dataset_manager
    meta = dm.get_metadata()

    # Dynamic dataset metrics display
    c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
    with c_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Samples</div>
            <div class="metric-value">{meta['total_samples']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Features</div>
            <div class="metric-value">{meta['num_features']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Training Set</div>
            <div class="metric-value">{meta['train_samples']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Test Set</div>
            <div class="metric-value">{meta['test_samples']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c_m5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Architecture</div>
            <div class="metric-value" style="font-size: 1.1rem; padding-top: 6px;">30→16→8→1</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_nn_ctrl, col_nn_dash = st.columns([1, 2.5], gap="medium")

    with col_nn_ctrl:
        st.markdown("#### ⚙️ Training Configuration")
        
        nn_selected_opts = st.multiselect(
            "Select Optimizers to Train",
            options=all_opts,
            default=["SGD", "Momentum", "AdaGrad", "RMSProp", "Adam", "AdamW"],
            help="Select optimizers for benchmark training",
            key="nn_selected_opts_multiselect"
        )

        nn_lr = st.number_input(
            "Learning Rate (η)",
            min_value=0.0001,
            max_value=1.0,
            value=0.01,
            step=0.005,
            format="%.4f",
            key="nn_lr_input"
        )

        nn_epochs = st.slider("Epochs", min_value=10, max_value=200, value=80, step=10, key="nn_epochs_slider")
        nn_batch_size = st.selectbox("Batch Size", options=[16, 32, 64, meta['train_samples']], index=1, format_func=lambda x: f"{x} (Full Batch)" if x == meta['train_samples'] else str(x), key="nn_batch_size_select")

        with st.expander("Optimizer Specific Hyperparameters", expanded=False):
            nn_beta1 = st.slider("Momentum / Adam β₁", 0.0, 0.99, 0.9, 0.05, key="nn_beta1_slider")
            nn_beta2 = st.slider("Adam / AdamW β₂", 0.5, 0.9999, 0.999, 0.001, format="%.4f", key="nn_beta2_slider")
            nn_wd = st.number_input("AdamW Weight Decay (λ)", 0.0, 0.1, 0.001, 0.001, format="%.4f", key="nn_wd_input")

        train_btn = st.button("🚀 Start Training", type="primary", use_container_width=True, key="nn_start_train_btn")

    with col_nn_dash:
        st.markdown("#### 📊 Live Training Dashboard")

        # Container placeholders for live updates during training
        plot_loss_holder = st.empty()
        plot_acc_holder = st.empty()
        plot_eff_lr_holder = st.empty()

        engine = NNTrainingEngine(dm)

        if train_btn:
            if not nn_selected_opts:
                st.error("Please select at least one optimizer to train.")
            else:
                histories: Dict[str, TrainingHistory] = {}
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                opt_configs = {}
                for name in nn_selected_opts:
                    if name == "SGD":
                        opt_configs[name] = {"lr": nn_lr}
                    elif name in ["Momentum", "NAG"]:
                        opt_configs[name] = {"lr": nn_lr, "beta": nn_beta1}
                    elif name == "AdaGrad":
                        opt_configs[name] = {"lr": nn_lr}
                    elif name == "RMSProp":
                        opt_configs[name] = {"lr": nn_lr, "beta": nn_beta1}
                    elif name == "Adam":
                        opt_configs[name] = {"lr": nn_lr, "beta1": nn_beta1, "beta2": nn_beta2}
                    elif name == "AdamW":
                        opt_configs[name] = {"lr": nn_lr, "beta1": nn_beta1, "beta2": nn_beta2, "weight_decay": nn_wd}

                total_runs = len(nn_selected_opts)
                for opt_idx, (opt_name, cfg) in enumerate(opt_configs.items()):
                    opt_inst = get_optimizer(opt_name, **cfg)

                    def make_cb(current_opt_name: str, opt_i: int):
                        def cb(ep: int, hist: TrainingHistory):
                            histories[current_opt_name] = hist
                            overall_prog = (opt_i + (ep / nn_epochs)) / total_runs
                            progress_bar.progress(min(overall_prog, 1.0))
                            status_text.text(f"Training {current_opt_name}... Epoch {ep}/{nn_epochs}")
                            
                            # Live redraw every 2 epochs or final epoch
                            if ep % 2 == 0 or ep == nn_epochs:
                                plot_loss_holder.plotly_chart(create_nn_loss_figure(histories), use_container_width=True)
                                plot_acc_holder.plotly_chart(create_nn_accuracy_figure(histories), use_container_width=True)
                                plot_eff_lr_holder.plotly_chart(create_effective_lr_figure(histories), use_container_width=True)
                        return cb

                    hist = engine.train_single_optimizer(
                        optimizer=opt_inst,
                        epochs=nn_epochs,
                        batch_size=nn_batch_size,
                        initial_seed=42,
                        epoch_callback=make_cb(opt_name, opt_idx)
                    )
                    histories[opt_name] = hist

                st.session_state.nn_histories = histories
                progress_bar.progress(1.0)
                status_text.success("Training completed successfully for all selected optimizers.")
        elif st.session_state.nn_histories:
            # Populate charts from session state when not actively training in current script run
            h = st.session_state.nn_histories
            plot_loss_holder.plotly_chart(create_nn_loss_figure(h), use_container_width=True)
            plot_acc_holder.plotly_chart(create_nn_accuracy_figure(h), use_container_width=True)
            plot_eff_lr_holder.plotly_chart(create_effective_lr_figure(h), use_container_width=True)
        else:
            st.info("Click **'🚀 Start Training'** above to run the neural network benchmark live.")

        if st.session_state.nn_histories:
            h = st.session_state.nn_histories
            st.markdown("### 📋 Automatic Comparison Table (PDF Section B3)")
            st.markdown("Convergence epoch is auto-computed as the first epoch where validation loss reaches within **1% of its final value**.")
            
            comp_df = NNTrainingEngine.generate_comparison_dataframe(h)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Click **'🚀 Start Training'** above to run the neural network benchmark live.")


# ==============================================================================
# TAB 3: CONDITIONING & LR SENSITIVITY EXPLORERS
# ==============================================================================
with tabs[2]:
    st.markdown("### 🔬 Conditioning & Learning-Rate Sensitivity Explorers")
    
    col_exp1, col_exp2 = st.columns(2, gap="large")

    with col_exp1:
        st.markdown("#### 1. Conditioning Explorer (PDF Section A5)")
        st.markdown(r"""
        The condition number $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$ of the Hessian matrix measures how elongated the loss bowl is:
        - **$L_1 = x^2 + 10y^2$** ($\kappa = 10$): Mild anisotropy.
        - **$L_2 = x^2 + 50y^2$** ($\kappa = 50$): Moderate elongation.
        - **$L_3 = x^2 + 100y^2$** ($\kappa = 100$): Steep narrow canyon.
        - **$L_4 = x^2 + 1000y^2$** ($\kappa = 1000$): Ill-conditioned valley.
        """)

        cond_opt = st.selectbox("Select Optimizer for Conditioning Analysis", ["SGD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam", "AdamW"], index=0, key="cond_opt_select")
        cond_lr = st.slider("Learning Rate for Conditioning Test", 0.001, 0.05, 0.01, 0.001, format="%.3f", key="cond_lr_slider")

        cond_trajs = {}
        for s_name, surf in SURFACES.items():
            opt_obj = get_optimizer(cond_opt, lr=cond_lr)
            t_pts, _ = surf.simulate_trajectory(opt_obj, start_point=(8.0, 8.0), max_iters=150)
            cond_trajs[f"{surf.name} (κ={surf.condition_number:.0f})"] = t_pts

        fig_cond = go.Figure()
        for s_label, pts in cond_trajs.items():
            fig_cond.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines+markers", name=s_label, marker=dict(size=4)))
        fig_cond.update_layout(
            title=f"<b>{cond_opt} Trajectories across Conditioning Levels</b>",
            xaxis_title="Parameter x",
            yaxis_title="Parameter y",
            template="plotly_dark",
            height=400,
            margin=dict(l=30, r=20, t=40, b=30)
        )
        st.plotly_chart(fig_cond, use_container_width=True, key="fig_cond_chart")

        st.caption("On $L_4$, SGD exhibits pronounced vertical oscillation because the gradient magnitude along y is substantially larger than along x.")

    with col_exp2:
        st.markdown("#### 2. Learning-Rate Sensitivity Explorer (PDF Section A6)")
        st.markdown(r"""
        Sweep learning rates across three canonical regimes:
        - **$\eta = 0.001$**: Lower step size, smooth steady convergence.
        - **$\eta = 0.01$**: Standard baseline learning rate.
        - **$\eta = 0.1$**: Higher step size, highlights oscillation or divergence boundaries.
        """)

        sens_opt = st.selectbox("Select Optimizer for Sensitivity Analysis", ["SGD", "Momentum", "AdaGrad", "RMSProp", "Adam"], index=0, key="sens_opt_select")
        sens_surf = SURFACES[DEFAULT_SURFACE_KEY]

        fig_sens = go.Figure()
        lr_presets = [0.001, 0.01, 0.1]
        colors = ["#2A9D8F", "#E76F51", "#E63946"]

        for idx, test_lr in enumerate(lr_presets):
            o = get_optimizer(sens_opt, lr=test_lr)
            t_pts, l_pts = sens_surf.simulate_trajectory(o, start_point=(8.0, 8.0), max_iters=100)
            fig_sens.add_trace(go.Scatter(
                x=list(range(len(l_pts))),
                y=l_pts,
                mode="lines",
                line=dict(color=colors[idx], width=2.5),
                name=f"η = {test_lr} ({'Diverged' if np.isnan(l_pts[-1]) or l_pts[-1] > 1e4 else 'Converged'})"
            ))

        fig_sens.update_layout(
            title=f"<b>{sens_opt} Loss Sensitivity across η ∈ {{0.001, 0.01, 0.1}}</b>",
            xaxis_title="Iteration t",
            yaxis_title="Loss L(θₜ) (Log Scale)",
            yaxis_type="log",
            template="plotly_dark",
            height=400,
            margin=dict(l=30, r=20, t=40, b=30)
        )
        st.plotly_chart(fig_sens, use_container_width=True, key="fig_sens_chart")
        st.caption("At η = 0.1 on the default bowl (where the maximum Hessian eigenvalue is 100), SGD exceeds the numerical stability threshold (η > 2/λ_max = 0.02).")


# ==============================================================================
# TAB 4: EXPLAIN-AS-YOU-GO PANEL
# ==============================================================================
with tabs[3]:
    st.markdown("### 💡 Explain-As-You-Go: Intuitive Mathematical Deep Dives")
    st.markdown("Detailed educational breakdowns of advanced optimizer mechanics based on assignment theory.")

    exp_nag, exp_adagrad, exp_rmsprop, exp_adamw = st.tabs([
        "🏎️ NAG Look-Ahead",
        "📊 AdaGrad Scaling",
        "🌊 RMSProp Moving Average",
        "⚖️ AdamW Decoupled Decay"
    ])

    with exp_nag:
        st.markdown(r"""
        #### Nesterov Accelerated Gradient (NAG)
        **Core Update Rule:**
        $$\theta_{\text{lookahead}} = \theta_t - \beta v_{t-1}$$
        $$v_t = \beta v_{t-1} + (1 - \beta) \nabla L(\theta_{\text{lookahead}})$$
        $$\theta_{t+1} = \theta_t - \eta v_t$$

        **Why Look-Ahead Reduces Overshoot:**
        - Standard Momentum computes the gradient at the *current* point $\theta_t$ and adds accumulated velocity blindly, which can overshoot valleys before correcting.
        - NAG evaluates the gradient at the look-ahead point $\theta_t - \beta v_{t-1}$. If the momentum is carrying the parameters up the opposite slope, the look-ahead gradient detects the upward slope in advance and applies an opposing corrective force before the full update is applied.
        """)

    with exp_adagrad:
        st.markdown(r"""
        #### AdaGrad (Adaptive Gradient Algorithm)
        **Core Update Rule:**
        $$G_t = G_{t-1} + g_t^2$$
        $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

        **Why Parameter-Wise Adaptive Scaling Works:**
        - In anisotropic loss landscapes (such as $L(x, y) = x^2 + 50y^2$), gradients along $y$ are substantially larger ($100y$) than along $x$ ($2x$).
        - AdaGrad accumulates squared gradients independently for every coordinate:
          $$G_{t, y} \gg G_{t, x} \implies \frac{\eta}{\sqrt{G_{t, y}}} \ll \frac{\eta}{\sqrt{G_{t, x}}}$$
        - This dynamically dampens the step size along the steep $y$-axis while maintaining a relatively larger effective learning rate along the shallow $x$-axis.
        """)

    with exp_rmsprop:
        st.markdown(r"""
        #### RMSProp (Root Mean Square Propagation)
        **Core Update Rule:**
        $$v_t = \beta v_{t-1} + (1 - \beta) g_t^2$$
        $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \odot g_t$$

        **How It Fixes AdaGrad's Diminishing Learning Rate:**
        - AdaGrad continuously accumulates positive squared gradients $g_t^2$ into $G_t$, causing the effective learning rate $\frac{\eta}{\sqrt{G_t}}$ to shrink toward zero over extended iterations.
        - RMSProp replaces the monotonic sum with an **exponential moving average** (discount factor $\beta=0.9$), discounting distant past gradients. The effective step size stabilizes according to recent local gradient magnitudes rather than decaying indefinitely.
        """)

    with exp_adamw:
        st.markdown(r"""
        #### AdamW (Adam with Decoupled Weight Decay)
        **Core Update Rule:**
        $$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right) = \theta_t (1 - \eta \lambda) - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

        **Decoupled Weight Decay vs. L2 Regularization in Gradient:**
        - In standard Adam with L2 regularization, weight decay is folded directly into the gradient: $g_t' = g_t + \lambda \theta_t$.
        - When scaled by the adaptive second-moment denominator $\sqrt{\hat{v}_t}$, parameters with large historical gradients experience reduced weight decay penalties, whereas parameters with small gradients receive disproportionately large penalties.
        - **AdamW decouples weight decay completely from the gradient step**, applying proportional shrinkage directly to $\theta_t$, ensuring uniform parameter regularization regardless of gradient magnitude.
        """)


# ==============================================================================
# TAB 5: HOW TO USE THIS TOOL
# ==============================================================================
with tabs[4]:
    st.markdown("### 📖 How to Use This Tool")
    
    col_h1, col_h2 = st.columns(2, gap="large")

    with col_h1:
        st.markdown(r"""
        #### Part A — 2D Optimizer Playground
        1. **Select Loss Surface:** Choose between $L_1$ to $L_4$ to observe how Hessian conditioning ($\kappa = 10 \to 1000$) alters trajectory paths.
        2. **Choose Optimizers:** Overlay multiple optimizers simultaneously to compare trajectories under identical conditions.
        3. **Tune Hyperparameters:** Adjust the learning rate ($\eta$), momentum factors ($\beta$), Adam moments ($\beta_1, \beta_2$), and AdamW weight decay ($\lambda$).
        4. **Animation Controls:** Use **Play**, **Pause**, **Step**, and **Reset** along with the speed slider to watch step-by-step optimization dynamics.
        5. **Compare Dual Views:** Observe parameter trajectories on the 2D contour map (View 1) alongside the synchronized loss curve (View 2).
        """)

    with col_h2:
        st.markdown(r"""
        #### Part B — Real Neural Network Training
        1. **Inspect Dataset Metrics:** Check dynamic sample counts (569 total, 30 features, train/test split) loaded from the Breast Cancer Wisconsin dataset.
        2. **Configure Benchmark:** Select optimizers, learning rate, batch size, and epoch count.
        3. **Start Training:** Click **🚀 Start Training** to run training from scratch using pure NumPy forward and backward passes.
        4. **Live Dashboard:** Monitor live training/test loss, accuracy, and the **effective learning rate ($\eta_{\text{eff}}$)** for representative weight $W_1[0,0]$.
        5. **Comparison Table:** Review auto-computed convergence epochs (first epoch reaching within 1% of final loss) and final test accuracies.
        """)

    st.markdown("---")
    st.markdown("""
    #### 📚 Optimizer Summary Reference
    | Optimizer | Key Innovation | Typical Use Case / Behavior |
    |---|---|---|
    | **SGD** | Direct first-order gradient descent | Baseline; sensitive to learning rate and curvature |
    | **Momentum** | Exponentially weighted velocity buffer | Accelerates along consistent slopes, dampens oscillation |
    | **NAG** | Look-ahead gradient evaluation | Proactively brakes before steep ascents |
    | **AdaGrad** | Coordinate-wise cumulative squared gradients | Adapts to parameter scale; effective rate shrinks over long horizons |
    | **RMSProp** | Exponential moving average of squared gradients | Maintains adaptive scaling without indefinite shrinkage |
    | **Adam** | Combines Momentum and RMSProp + bias correction | Robust across diverse deep learning architectures |
    | **AdamW** | Decoupled weight decay regularization | Prevents regularization distortion in adaptive optimization |
    """)
