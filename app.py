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
    create_effective_lr_figure,
    PLOTLY_CONFIG
)

# Page Configuration
st.set_page_config(
    page_title="Deep Learning Optimizer Visualizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling for Academic Presentation & High Contrast
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        color: var(--text-color, #1F2328);
    }
    .sub-title {
        font-size: 1.0rem;
        color: var(--text-color-secondary, #57606A);
        margin-bottom: 1.2rem;
    }
    @media (prefers-color-scheme: light) {
        .main-title { color: #1F2328 !important; }
        .sub-title { color: #57606A !important; }
    }
    @media (prefers-color-scheme: dark) {
        .main-title { color: #F0F6FC !important; }
        .sub-title { color: #8B949E !important; }
    }
    .metric-card {
        background-color: rgba(127, 127, 127, 0.08);
        border: 1px solid rgba(127, 127, 127, 0.2);
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
    }
    .metric-title {
        font-size: 0.8rem;
        color: var(--text-color-secondary, #8B949E);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #58A6FF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 8px;
        padding-bottom: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "part_a_iteration_slider" not in st.session_state:
    st.session_state.part_a_iteration_slider = 0
if "play_advance" not in st.session_state:
    st.session_state.play_advance = 0
if "part_a_max_limit" not in st.session_state:
    st.session_state.part_a_max_limit = 500
if "dataset_manager" not in st.session_state:
    st.session_state.dataset_manager = DatasetManager(test_size=0.2, random_state=42)
if "nn_histories" not in st.session_state:
    st.session_state.nn_histories = {}
if "nn_is_training" not in st.session_state:
    st.session_state.nn_is_training = False

# Advance iteration before widget instantiation when playing
if st.session_state.is_playing and st.session_state.play_advance > 0:
    st.session_state.part_a_iteration_slider = min(
        st.session_state.part_a_iteration_slider + st.session_state.play_advance,
        st.session_state.part_a_max_limit
    )
    st.session_state.play_advance = 0


# Title & Header
st.markdown("<div class='main-title'>Optimizer Visualizer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Explore how seven optimization algorithms behave on mathematical loss surfaces and during neural-network training.</div>", unsafe_allow_html=True)


# Main Tabs Navigation
tabs = st.tabs([
    "Part A — 2D Playground",
    "Part B — Neural Network",
    "Conditioning & LR Sensitivity",
    "Optimizer Explanations",
    "How to Use"
])


# ==============================================================================
# TAB 1: PART A — 2D OPTIMIZER PLAYGROUND
# ==============================================================================
with tabs[0]:
    col_ctrl, col_viz = st.columns([1, 2.5], gap="medium")

    with col_ctrl:
        st.markdown("#### Surface & Optimizers")
        
        # 1. Loss surface selector
        surface_key = st.selectbox(
            "Loss Surface",
            options=list(SURFACES.keys()),
            index=list(SURFACES.keys()).index(DEFAULT_SURFACE_KEY),
            help="Select a 2D surface to test optimizer paths under different conditioning levels",
            key="part_a_surface_select"
        )
        selected_surface = get_surface(surface_key)

        st.caption(f"**Formula:** $L(x, y) = {selected_surface.formula_str}$ | **Condition Number (κ):** {selected_surface.condition_number:.0f} | **Minimum:** (0, 0)")

        # 2. Multi-select optimizers
        all_opts = ["SGD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam", "AdamW"]
        selected_opts = st.multiselect(
            "Selected Optimizers",
            options=all_opts,
            default=["SGD", "Momentum", "AdaGrad", "Adam"],
            help="Select one or more optimizers to compare simultaneously",
            key="part_a_optimizer_multiselect"
        )

        if not selected_opts:
            st.warning("Please select at least one optimizer.")
            selected_opts = ["Adam"]

        st.markdown("#### Parameters")
        
        # Learning rate — log-spaced presets for precise control
        _lr_presets_a = [0.0001, 0.0003, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0]
        lr = st.select_slider(
            "Learning Rate (η)",
            options=_lr_presets_a,
            value=0.01,
            format_func=lambda v: f"{v:.4g}",
            help="Controls step size along the gradient vector (log-spaced presets).",
            key="part_a_lr_input"
        )

        # Relevant optimizer-specific hyperparameters
        has_momentum = any(o in selected_opts for o in ["Momentum", "NAG"])
        has_rmsprop = "RMSProp" in selected_opts
        has_adam = any(o in selected_opts for o in ["Adam", "AdamW"])
        has_adamw = "AdamW" in selected_opts

        beta_momentum = 0.9
        beta_rmsprop = 0.9
        beta1_adam = 0.9
        beta2_adam = 0.999
        weight_decay = 0.001

        if has_momentum or has_rmsprop or has_adam or has_adamw:
            with st.expander("Hyperparameter Settings", expanded=False):
                if has_momentum:
                    beta_momentum = st.slider("Momentum / NAG β", 0.0, 0.99, 0.9, 0.05, key="part_a_beta_momentum")
                if has_rmsprop:
                    beta_rmsprop = st.slider("RMSProp β", 0.0, 0.99, 0.9, 0.05, key="part_a_beta_rmsprop")
                if has_adam:
                    beta1_adam = st.slider("Adam/AdamW β₁", 0.0, 0.99, 0.9, 0.05, key="part_a_beta1_adam")
                    beta2_adam = st.slider("Adam/AdamW β₂", 0.5, 0.9999, 0.999, 0.001, format="%.4f", key="part_a_beta2_adam")
                if has_adamw:
                    weight_decay = st.number_input("AdamW Weight Decay (λ)", 0.0, 0.5, 0.001, 0.001, format="%.4f", key="part_a_lambda_adamw")

        # Starting point
        st.markdown("#### Starting Coordinates & Steps")
        col_x0, col_y0 = st.columns(2)
        with col_x0:
            x0 = st.number_input("x₀", value=8.0, step=0.5, key="part_a_x0_input")
        with col_y0:
            y0 = st.number_input("y₀", value=8.0, step=0.5, key="part_a_y0_input")

        max_steps = st.slider("Max Iterations", min_value=50, max_value=500, value=300, step=25, key="part_a_max_steps_slider")

        # Detect parameter modifications to reset animation state cleanly
        curr_params = (
            surface_key,
            tuple(sorted(selected_opts)),
            float(lr),
            float(beta_momentum),
            float(beta_rmsprop),
            float(beta1_adam),
            float(beta2_adam),
            float(weight_decay),
            float(x0),
            float(y0),
            int(max_steps)
        )
        if "part_a_prev_params" not in st.session_state or st.session_state.part_a_prev_params != curr_params:
            st.session_state.part_a_prev_params = curr_params
            st.session_state.part_a_iteration_slider = 0
            st.session_state.is_playing = False
            st.session_state.play_advance = 0

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

        actual_max_len = max([len(t) for t in trajectories.values()]) if trajectories else max_steps
        max_valid_step = max(actual_max_len - 1, 1)
        st.session_state.part_a_max_limit = max_valid_step

        # Button callbacks
        def on_step_clicked():
            st.session_state.is_playing = False
            max_limit = st.session_state.get("part_a_max_limit", 300)
            st.session_state.part_a_iteration_slider = min(
                st.session_state.get("part_a_iteration_slider", 0) + 1,
                max_limit
            )

        def on_reset_clicked():
            st.session_state.is_playing = False
            st.session_state.part_a_iteration_slider = 0
            st.session_state.play_advance = 0

        def on_play_clicked():
            st.session_state.is_playing = True

        def on_pause_clicked():
            st.session_state.is_playing = False
            st.session_state.play_advance = 0

        # Animation Controls
        st.markdown("#### Animation Controls")
        c_play, c_pause, c_step, c_reset = st.columns(4)
        
        with c_play:
            st.button("Play", width="stretch", key="part_a_play", on_click=on_play_clicked)
        with c_pause:
            st.button("Pause", width="stretch", key="part_a_pause", on_click=on_pause_clicked)
        with c_step:
            st.button("Step", width="stretch", key="part_a_step", on_click=on_step_clicked)
        with c_reset:
            st.button("Reset", width="stretch", key="part_a_reset", on_click=on_reset_clicked)

        anim_speed = st.select_slider(
            "Animation Speed",
            options=["1x", "2x", "5x", "Max"],
            value="2x",
            key="part_a_anim_speed"
        )
        step_increment = {"1x": 1, "2x": 2, "5x": 5, "Max": 15}[anim_speed]

        # Slider for manual scrubbing and step inspection
        st.slider(
            "Iteration",
            min_value=0,
            max_value=max_valid_step,
            step=1,
            key="part_a_iteration_slider"
        )
        current_step = st.session_state.part_a_iteration_slider

    with col_viz:
        st.markdown("#### Synchronized Trajectory & Loss Views")
        v1_col, v2_col = st.columns(2, gap="small")
        
        with v1_col:
            fig_contour = create_contour_figure(
                surface=selected_surface,
                trajectories=trajectories,
                current_step=current_step,
                start_point=(x0, y0),
                x_range=(-max(abs(x0) + 2, 10), max(abs(x0) + 2, 10)),
                y_range=(-max(abs(y0) + 2, 10), max(abs(y0) + 2, 10))
            )
            st.plotly_chart(fig_contour, key="fig_contour_chart", config=PLOTLY_CONFIG)

        with v2_col:
            fig_loss = create_loss_curve_figure(
                losses_dict=losses_dict,
                current_step=current_step,
                log_scale=True
            )
            st.plotly_chart(fig_loss, key="fig_loss_chart", config=PLOTLY_CONFIG)

        # Step Status Metrics
        st.markdown(f"##### Current Optimizer Position & Loss (Iteration {current_step}/{max_valid_step})")
        m_cols = st.columns(len(selected_opts))
        for idx, opt_name in enumerate(selected_opts):
            with m_cols[idx]:
                traj = trajectories[opt_name]
                step_idx = min(current_step, len(traj) - 1)
                curr_pos = traj[step_idx]
                curr_loss = losses_dict[opt_name][step_idx]
                st.markdown(f"""
                <div style="border-left: 3px solid {OPTIMIZER_COLORS[opt_name]}; padding-left: 8px; margin-bottom: 8px;">
                    <strong>{opt_name}</strong><br>
                    <small>({curr_pos[0]:.2f}, {curr_pos[1]:.2f})</small><br>
                    <small>Loss: <b>{curr_loss:.4e}</b></small>
                </div>
                """, unsafe_allow_html=True)

    # Automatic animation loop step
    if st.session_state.is_playing:
        if current_step < max_valid_step:
            sleep_times = {"1x": 0.12, "2x": 0.07, "5x": 0.03, "Max": 0.01}
            time.sleep(sleep_times.get(anim_speed, 0.05))
            st.session_state.play_advance = step_increment
            st.rerun()
        else:
            st.session_state.is_playing = False
            st.session_state.play_advance = 0


# ==============================================================================
# TAB 2: PART B — REAL NEURAL NETWORK TRAINING
# ==============================================================================
with tabs[1]:
    st.markdown("#### Multi-Layer Perceptron on Breast Cancer Wisconsin Dataset")
    st.caption("Architecture: 30 → 16 (ReLU) → 8 (ReLU) → 1 (Sigmoid). Implemented from scratch in NumPy with analytical backpropagation.")

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
            <div class="metric-title">Layers</div>
            <div class="metric-value" style="font-size: 1.1rem; padding-top: 6px;">30→16→8→1</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_nn_ctrl, col_nn_dash = st.columns([1, 2.5], gap="medium")

    with col_nn_ctrl:
        st.markdown("#### Training Setup")
        
        nn_selected_opts = st.multiselect(
            "Optimizers to Train",
            options=all_opts,
            default=["SGD", "Momentum", "AdaGrad", "RMSProp", "Adam", "AdamW"],
            help="Select optimizers to benchmark",
            key="nn_selected_opts_multiselect"
        )

        _lr_presets_b = [0.0001, 0.0003, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0]
        nn_lr = st.select_slider(
            "Learning Rate (η)",
            options=_lr_presets_b,
            value=0.01,
            format_func=lambda v: f"{v:.4g}",
            key="nn_lr_input"
        )

        nn_epochs = st.slider("Epochs", min_value=10, max_value=200, value=80, step=10, key="nn_epochs_slider")
        nn_batch_size = st.selectbox("Batch Size", options=[16, 32, 64, meta['train_samples']], index=1, format_func=lambda x: f"{x} (Full Batch)" if x == meta['train_samples'] else str(x), key="nn_batch_size_select")

        with st.expander("Hyperparameter Options", expanded=False):
            nn_beta1 = st.slider("Momentum / Adam β₁", 0.0, 0.99, 0.9, 0.05, key="nn_beta1_slider")
            nn_beta2 = st.slider("Adam / AdamW β₂", 0.5, 0.9999, 0.999, 0.001, format="%.4f", key="nn_beta2_slider")
            nn_wd = st.number_input("AdamW Weight Decay (λ)", 0.0, 0.1, 0.001, 0.001, format="%.4f", key="nn_wd_input")

        train_btn = st.button("Start Training", type="primary", width="stretch", key="nn_start_train_btn")

    with col_nn_dash:
        st.markdown("#### Live Training Dashboard")

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
                            
                            if ep % 2 == 0 or ep == nn_epochs:
                                plot_loss_holder.plotly_chart(create_nn_loss_figure(histories), key=f"nn_loss_live_{current_opt_name}_{ep}", config=PLOTLY_CONFIG)
                                plot_acc_holder.plotly_chart(create_nn_accuracy_figure(histories), key=f"nn_acc_live_{current_opt_name}_{ep}", config=PLOTLY_CONFIG)
                                plot_eff_lr_holder.plotly_chart(create_effective_lr_figure(histories), key=f"nn_lr_live_{current_opt_name}_{ep}", config=PLOTLY_CONFIG)
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
                status_text.success("Training completed successfully.")
        elif st.session_state.nn_histories:
            h = st.session_state.nn_histories
            plot_loss_holder.plotly_chart(create_nn_loss_figure(h), key="nn_loss_static", config=PLOTLY_CONFIG)
            plot_acc_holder.plotly_chart(create_nn_accuracy_figure(h), key="nn_acc_static", config=PLOTLY_CONFIG)
            plot_eff_lr_holder.plotly_chart(create_effective_lr_figure(h), key="nn_eff_lr_static", config=PLOTLY_CONFIG)
        else:
            st.info("Click 'Start Training' above to run the neural network training benchmark.")

        if st.session_state.nn_histories:
            h = st.session_state.nn_histories
            st.markdown("#### Comparison Table")
            st.caption("Convergence epoch is defined as the first epoch at which validation loss reaches within 1% of its final value.")
            
            comp_df = NNTrainingEngine.generate_comparison_dataframe(h)
            st.dataframe(comp_df, hide_index=True)


# ==============================================================================
# TAB 3: CONDITIONING & LR SENSITIVITY EXPLORERS
# ==============================================================================
with tabs[2]:
    st.markdown("#### Explorers: Conditioning & Learning-Rate Sensitivity")
    
    col_exp1, col_exp2 = st.columns(2, gap="large")

    with col_exp1:
        st.markdown("##### 1. Conditioning Explorer (PDF Section A5)")
        st.markdown(r"""
        The condition number $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$ of the Hessian matrix measures curvature anisotropy:
        - **$L_1 = x^2 + 10y^2$** ($\kappa = 10$): Mild anisotropy.
        - **$L_2 = x^2 + 50y^2$** ($\kappa = 50$): Moderate elongation.
        - **$L_3 = x^2 + 100y^2$** ($\kappa = 100$): Steep narrow canyon.
        - **$L_4 = x^2 + 1000y^2$** ($\kappa = 1000$): Ill-conditioned valley.
        """)

        cond_opt = st.selectbox("Optimizer for Conditioning Comparison", ["SGD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam", "AdamW"], index=0, key="cond_opt_select")
        cond_lr = st.slider("Learning Rate", 0.001, 0.05, 0.01, 0.001, format="%.3f", key="cond_lr_slider")

        cond_trajs = {}
        for s_name, surf in SURFACES.items():
            opt_obj = get_optimizer(cond_opt, lr=cond_lr)
            t_pts, _ = surf.simulate_trajectory(opt_obj, start_point=(8.0, 8.0), max_iters=150)
            cond_trajs[f"{surf.name} (κ={surf.condition_number:.0f})"] = t_pts

        fig_cond = go.Figure()
        for s_label, pts in cond_trajs.items():
            fig_cond.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines+markers", name=s_label, marker=dict(size=4)))
        fig_cond.update_layout(
            title=dict(
                text=f"<b>{cond_opt} Trajectories across Conditioning Levels</b>",
                x=0.5,
                xanchor="center",
                y=0.98,
                yanchor="top",
                font=dict(size=13),
                pad=dict(r=70)
            ),
            xaxis_title="Parameter x",
            yaxis_title="Parameter y",
            template="plotly_dark",
            height=430,
            margin=dict(l=40, r=20, t=100, b=35),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.88,
                xanchor="center",
                x=0.5,
                font=dict(size=9.5),
                bgcolor="rgba(0,0,0,0)"
            ),
            modebar=dict(
                orientation="h",
                bgcolor="rgba(0,0,0,0)",
                activecolor="#58A6FF"
            )
        )
        st.plotly_chart(fig_cond, key="fig_cond_chart", config=PLOTLY_CONFIG)

        st.caption("On $L_4$, SGD shows larger vertical oscillations because the gradient magnitude along y is substantially larger than along x.")

    with col_exp2:
        st.markdown("##### 2. Learning-Rate Sensitivity Explorer (PDF Section A6)")
        st.markdown(r"""
        Examine how step size affects convergence on the default bowl ($L_2 = x^2 + 50y^2$):
        - **$\eta = 0.001$**: Smaller step size, smooth steady progress.
        - **$\eta = 0.01$**: Default baseline learning rate.
        - **$\eta = 0.1$**: Larger step size, demonstrates oscillation or divergence boundaries.
        """)

        sens_opt = st.selectbox("Optimizer for Sensitivity Test", ["SGD", "Momentum", "AdaGrad", "RMSProp", "Adam"], index=0, key="sens_opt_select")
        sens_surf = SURFACES[DEFAULT_SURFACE_KEY]

        fig_sens = go.Figure()
        lr_presets = [0.001, 0.01, 0.1]
        colors = ["#2A9D8F", "#E76F51", "#E63946"]

        for idx, test_lr in enumerate(lr_presets):
            o = get_optimizer(sens_opt, lr=test_lr)
            t_pts, l_pts = sens_surf.simulate_trajectory(o, start_point=(8.0, 8.0), max_iters=100)
            status_label = "Diverged" if (np.isnan(l_pts[-1]) or l_pts[-1] > 1e4) else "Converged"
            fig_sens.add_trace(go.Scatter(
                x=list(range(len(l_pts))),
                y=l_pts,
                mode="lines",
                line=dict(color=colors[idx], width=2.5),
                name=f"η = {test_lr} ({status_label})"
            ))

        fig_sens.update_layout(
            title=dict(
                text=f"<b>{sens_opt} Loss Sensitivity across η ∈ {{0.001, 0.01, 0.1}}</b>",
                x=0.5,
                xanchor="center",
                y=0.98,
                yanchor="top",
                font=dict(size=13),
                pad=dict(r=70)
            ),
            xaxis_title="Iteration t",
            yaxis_title="Loss L(θₜ) (Log Scale)",
            yaxis_type="log",
            template="plotly_dark",
            height=430,
            margin=dict(l=40, r=20, t=100, b=35),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.88,
                xanchor="center",
                x=0.5,
                font=dict(size=9.5),
                bgcolor="rgba(0,0,0,0)"
            ),
            modebar=dict(
                orientation="h",
                bgcolor="rgba(0,0,0,0)",
                activecolor="#58A6FF"
            )
        )
        st.plotly_chart(fig_sens, key="fig_sens_chart", config=PLOTLY_CONFIG)
        st.caption("At η = 0.1 on the default bowl (where maximum Hessian eigenvalue is 100), constant-step gradient descent exceeds the stability bound (η > 2/λ_max = 0.02).")


# ==============================================================================
# TAB 4: EXPLAIN-AS-YOU-GO PANEL
# ==============================================================================
with tabs[3]:
    st.markdown("#### Optimizer Explanations")
    st.caption("Core mathematical update rules and intuitive design rationale based on the assignment.")

    exp_nag, exp_adagrad, exp_rmsprop, exp_adamw = st.tabs([
        "NAG Look-Ahead",
        "AdaGrad Scaling",
        "RMSProp Moving Average",
        "AdamW Decoupled Decay"
    ])

    with exp_nag:
        st.markdown(r"""
        ##### Nesterov Accelerated Gradient (NAG)
        **Update Rule:**
        $$\theta_{\text{lookahead}} = \theta_t - \eta \beta v_{t-1}$$
        $$v_t = \beta v_{t-1} + (1 - \beta) \nabla L(\theta_{\text{lookahead}})$$
        $$\theta_{t+1} = \theta_t - \eta v_t$$

        > **Implementation note:** The PDF shorthand writes $\theta_t - \beta v_{t-1}$, but because
        > $v$ is an EMA of gradients (gradient units) and $\theta$ lives in parameter space, the
        > lookahead displacement must be scaled by $\eta$ — exactly as in the final update
        > $\theta_{t+1} = \theta_t - \eta v_t$ — to keep both steps dimensionally consistent.

        **Mechanism:**
        - Standard Momentum computes the gradient at the current position $\theta_t$ and adds accumulated velocity.
        - NAG evaluates the gradient at the look-ahead point $\theta_t - \eta\beta v_{t-1}$. If accumulated velocity is carrying parameters up an opposing slope, the look-ahead gradient detects the slope in advance and applies an opposing corrective force before the full update is applied.
        """)

    with exp_adagrad:
        st.markdown(r"""
        ##### AdaGrad (Adaptive Gradient Algorithm)
        **Update Rule:**
        $$G_t = G_{t-1} + g_t^2$$
        $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

        **Mechanism:**
        - In anisotropic loss landscapes (such as $L(x, y) = x^2 + 50y^2$), gradients along $y$ are substantially larger ($100y$) than along $x$ ($2x$).
        - AdaGrad accumulates squared gradients coordinate-by-coordinate:
          $$G_{t, y} \gg G_{t, x} \implies \frac{\eta}{\sqrt{G_{t, y}}} \ll \frac{\eta}{\sqrt{G_{t, x}}}$$
        - This scales down step sizes along the steep $y$-axis while maintaining a relatively larger effective learning rate along the shallow $x$-axis.
        """)

    with exp_rmsprop:
        st.markdown(r"""
        ##### RMSProp (Root Mean Square Propagation)
        **Update Rule:**
        $$v_t = \beta v_{t-1} + (1 - \beta) g_t^2$$
        $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \odot g_t$$

        **Mechanism:**
        - AdaGrad continuously accumulates positive squared gradients $g_t^2$ into $G_t$, causing the effective learning rate $\frac{\eta}{\sqrt{G_t}}$ to shrink toward zero over long runs.
        - RMSProp replaces the cumulative sum with an **exponential moving average** (discount factor $\beta=0.9$), discounting distant past gradients. The effective step size stabilizes according to recent gradient energy rather than decaying indefinitely.
        """)

    with exp_adamw:
        st.markdown(r"""
        ##### AdamW (Adam with Decoupled Weight Decay)
        **Update Rule:**
        $$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right) = \theta_t (1 - \eta \lambda) - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

        **Mechanism:**
        - In standard Adam with L2 regularization, weight decay is folded directly into the gradient: $g_t' = g_t + \lambda \theta_t$.
        - When scaled by the adaptive second-moment denominator $\sqrt{\hat{v}_t}$, parameters with large historical gradients experience reduced weight decay penalties, whereas parameters with small gradients receive disproportionately large penalties.
        - **AdamW decouples weight decay completely from the gradient step**, applying proportional shrinkage directly to $\theta_t$, ensuring uniform parameter regularization regardless of gradient magnitude.
        """)


# ==============================================================================
# TAB 5: HOW TO USE THIS TOOL
# ==============================================================================
with tabs[4]:
    st.markdown("#### How to Use This Tool")
    
    col_h1, col_h2 = st.columns(2, gap="large")

    with col_h1:
        st.markdown(r"""
        ##### Part A — 2D Optimizer Playground
        1. **Select Loss Surface:** Choose between $L_1$ to $L_4$ to observe how Hessian conditioning ($\kappa = 10 \to 1000$) alters trajectory paths.
        2. **Choose Optimizers:** Overlay multiple optimizers simultaneously to compare trajectories under identical conditions.
        3. **Tune Hyperparameters:** Adjust the learning rate ($\eta$), momentum factors ($\beta$), Adam moments ($\beta_1, \beta_2$), and AdamW weight decay ($\lambda$).
        4. **Animation Controls:** Use **Play**, **Pause**, **Step**, and **Reset** along with the speed slider to watch step-by-step optimization dynamics.
        5. **Compare Dual Views:** Observe parameter trajectories on the 2D contour map (View 1) alongside the synchronized loss curve (View 2).
        """)

    with col_h2:
        st.markdown(r"""
        ##### Part B — Real Neural Network Training
        1. **Inspect Dataset Metrics:** Check dynamic sample counts (569 total, 30 features, train/test split) loaded from the Breast Cancer Wisconsin dataset.
        2. **Configure Benchmark:** Select optimizers, learning rate, batch size, and epoch count.
        3. **Start Training:** Click **Start Training** to run training from scratch using pure NumPy forward and backward passes.
        4. **Live Dashboard:** Monitor live training/test loss, accuracy, and the **effective learning rate ($\eta_{\text{eff}}$)** for representative weight $W_1[0,0]$.
        5. **Comparison Table:** Review auto-computed convergence epochs (first epoch reaching within 1% of final loss) and final test accuracies.
        """)

    st.markdown("---")
    st.markdown("""
    ##### Optimizer Summary Reference
    | Optimizer | Key Idea | Primary Characteristic |
    |---|---|---|
    | **SGD** | First-order gradient descent | Direct gradient updates; sensitive to curvature and learning rate |
    | **Momentum** | Velocity buffer | Accumulates consistent directional updates, reduces oscillation |
    | **NAG** | Look-ahead gradient | Evaluates gradient at predicted position to reduce overshoot |
    | **AdaGrad** | Cumulative squared gradients | Parameter-wise scaling; learning rate shrinks over time |
    | **RMSProp** | Exponential moving average of squared gradients | Parameter-wise scaling with recent gradient memory |
    | **Adam** | First and second moments with bias correction | Combines momentum and adaptive scaling with early-step correction |
    | **AdamW** | Decoupled weight decay | Applies weight decay directly to parameters, preserving true regularization |
    """)
