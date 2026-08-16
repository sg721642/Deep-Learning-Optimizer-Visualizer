"""
From SGD to AdamW — Deep Learning Optimizer Visualizer
Visualization utilities for synchronized 2D contour maps, loss curves, and neural network dashboards.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go
from src.optimizers import OPTIMIZER_COLORS
from src.surfaces import LossSurface2D
from src.experiment import TrainingHistory

PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"]
}


def create_contour_figure(
    surface: LossSurface2D,
    trajectories: Dict[str, np.ndarray],
    current_step: int,
    start_point: Tuple[float, float] = (8.0, 8.0),
    x_range: Tuple[float, float] = (-10.0, 10.0),
    y_range: Tuple[float, float] = (-10.0, 10.0),
    grid_res: int = 120
) -> go.Figure:
    """
    Generate interactive Plotly contour map with animated optimizer trajectories.
    View 1 per PDF spec:
        - Filled contour plot of selected loss surface
        - Global minimum marked with ★ at (0, 0)
        - Growing connected path per optimizer
        - Current step marker
        - Distinct consistent color scheme
        - Bounded auto-scaling / divergence handling
    """
    # Determine sensible plotting bounds
    base_x = max(abs(start_point[0]) + 2.0, 10.0)
    base_y = max(abs(start_point[1]) + 2.0, 10.0)

    # Check active trajectory positions up to current step
    max_x_obs = base_x
    max_y_obs = base_y
    for opt_name, traj in trajectories.items():
        if len(traj) > 0:
            idx = min(current_step, len(traj) - 1)
            sub = traj[:idx + 1]
            max_x_obs = max(max_x_obs, float(np.nanmax(np.abs(sub[:, 0]))))
            max_y_obs = max(max_y_obs, float(np.nanmax(np.abs(sub[:, 1]))))

    # Bounded expansion to prevent the origin from becoming a microscopic dot
    plot_limit_x = min(max_x_obs * 1.15, 35.0)
    plot_limit_y = min(max_y_obs * 1.15, 35.0)
    plot_limit = max(plot_limit_x, plot_limit_y, 10.0)

    eff_x_range = (-plot_limit, plot_limit)
    eff_y_range = (-plot_limit, plot_limit)

    # Create coordinate grid
    x_vals = np.linspace(eff_x_range[0], eff_x_range[1], grid_res)
    y_vals = np.linspace(eff_y_range[0], eff_y_range[1], grid_res)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = surface.loss(X, Y)

    fig = go.Figure()

    # Base filled contour layer
    fig.add_trace(go.Contour(
        x=x_vals,
        y=y_vals,
        z=Z,
        colorscale="Viridis",
        contours=dict(
            coloring="heatmap",
            showlines=True,
        ),
        line=dict(width=0.5, color="rgba(255, 255, 255, 0.4)"),
        opacity=0.85,
        hoverinfo="x+y+z",
        showscale=False,
        name="Loss Surface"
    ))

    # Mark Global Minimum at (0, 0)
    fig.add_trace(go.Scatter(
        x=[0.0],
        y=[0.0],
        mode="markers+text",
        marker=dict(size=15, color="#FFD700", symbol="star", line=dict(color="#000", width=1.5)),
        text=["★ (0,0)"],
        textposition="bottom right",
        textfont=dict(color="#FFFFFF", size=10, family="sans-serif"),
        name="Global Min ★",
        showlegend=True
    ))

    # Mark Starting Point
    fig.add_trace(go.Scatter(
        x=[start_point[0]],
        y=[start_point[1]],
        mode="markers+text",
        marker=dict(size=11, color="#FFFFFF", symbol="circle-open", line=dict(color="#FFFFFF", width=2)),
        text=["Start (x₀,y₀)"],
        textposition="top right",
        textfont=dict(color="#FFFFFF", size=10),
        name="Start Point",
        showlegend=False
    ))

    # Plot each optimizer's trajectory up to current_step
    for opt_name, traj in trajectories.items():
        if len(traj) == 0:
            continue
        
        color = OPTIMIZER_COLORS.get(opt_name, "#FFFFFF")
        idx = min(current_step, len(traj) - 1)
        sub_traj = traj[:idx + 1]

        # Growing trajectory line (clamped to plot range for drawing stability)
        clamped_x = np.clip(sub_traj[:, 0], -plot_limit * 1.5, plot_limit * 1.5)
        clamped_y = np.clip(sub_traj[:, 1], -plot_limit * 1.5, plot_limit * 1.5)

        fig.add_trace(go.Scatter(
            x=clamped_x,
            y=clamped_y,
            mode="lines",
            line=dict(color=color, width=2.8),
            name=f"{opt_name}",
            legendgroup=opt_name,
            showlegend=True
        ))

        # Current position marker
        curr_x, curr_y = sub_traj[-1, 0], sub_traj[-1, 1]
        is_diverged = abs(curr_x) > plot_limit or abs(curr_y) > plot_limit
        display_x = np.clip(curr_x, -plot_limit * 0.96, plot_limit * 0.96)
        display_y = np.clip(curr_y, -plot_limit * 0.96, plot_limit * 0.96)

        marker_sym = "triangle-up" if is_diverged else "circle"
        hover_label = f"{opt_name}<br>Step {idx}<br>x={curr_x:.3f}, y={curr_y:.3f}<br>Loss={surface.loss(curr_x, curr_y):.4e}"
        if is_diverged:
            hover_label += "<br><b>[Diverged out of bounds]</b>"

        fig.add_trace(go.Scatter(
            x=[display_x],
            y=[display_y],
            mode="markers",
            marker=dict(size=10, color=color, symbol=marker_sym, line=dict(color="#FFFFFF", width=1.5)),
            name=f"{opt_name} (Current)",
            legendgroup=opt_name,
            showlegend=False,
            hovertext=hover_label,
            hoverinfo="text"
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>View 1: Parameter Trajectories on {surface.formula_str}</b> (Step {current_step}, κ={surface.condition_number:.0f})",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=13),
            pad=dict(r=70)
        ),
        xaxis=dict(
            title="Parameter x (Shallow Axis)",
            range=list(eff_x_range),
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.3)",
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title="Parameter y (Steep Axis)",
            range=list(eff_y_range),
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.3)",
            gridcolor="rgba(255,255,255,0.1)"
        ),
        margin=dict(l=45, r=25, t=100, b=45),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.88,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10)
        ),
        modebar=dict(
            orientation="h",
            bgcolor="rgba(0,0,0,0)",
            activecolor="#58A6FF"
        ),
        template="plotly_dark",
        height=530
    )
    return fig


def create_loss_curve_figure(
    losses_dict: Dict[str, np.ndarray],
    current_step: int,
    log_scale: bool = True
) -> go.Figure:
    """
    Generate interactive Plotly loss curve figure synchronized with View 1.
    View 2 per PDF spec:
        - L(θ_t) vs iteration t
        - Updating in lock-step with View 1
        - Shared color palette
    """
    fig = go.Figure()

    max_len = 1
    for opt_name, losses in losses_dict.items():
        if len(losses) == 0:
            continue
        max_len = max(max_len, len(losses))
        color = OPTIMIZER_COLORS.get(opt_name, "#FFFFFF")
        idx = min(current_step, len(losses) - 1)
        sub_losses = losses[:idx + 1]
        iterations = list(range(len(sub_losses)))

        # Trace for loss history
        fig.add_trace(go.Scatter(
            x=iterations,
            y=sub_losses,
            mode="lines",
            line=dict(color=color, width=2.5),
            name=opt_name,
            legendgroup=opt_name,
            showlegend=True
        ))

        # Marker at current step
        if len(sub_losses) > 0:
            curr_loss = sub_losses[-1]
            fig.add_trace(go.Scatter(
                x=[idx],
                y=[curr_loss],
                mode="markers",
                marker=dict(size=8, color=color, symbol="circle", line=dict(color="#FFFFFF", width=1.5)),
                name=f"{opt_name} Current",
                legendgroup=opt_name,
                showlegend=False,
                hovertext=f"{opt_name}<br>Step {idx}<br>Loss: {curr_loss:.6f}",
                hoverinfo="text"
            ))

    fig.update_layout(
        title=dict(
            text=f"<b>View 2: Loss Curve L(θₜ) vs. Iteration t</b> (Step {current_step})",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=13),
            pad=dict(r=70)
        ),
        xaxis=dict(
            title="Iteration t",
            range=[0, max(max_len, 50)],
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title="Loss L(θₜ)" + (" (Log Scale)" if log_scale else ""),
            type="log" if log_scale else "linear",
            gridcolor="rgba(255,255,255,0.1)"
        ),
        margin=dict(l=50, r=25, t=100, b=45),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.88,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10)
        ),
        modebar=dict(
            orientation="h",
            bgcolor="rgba(0,0,0,0)",
            activecolor="#58A6FF"
        ),
        template="plotly_dark",
        height=530
    )
    return fig


def create_nn_loss_figure(histories: Dict[str, TrainingHistory]) -> go.Figure:
    """Create training and validation loss curves for Part B live dashboard."""
    fig = go.Figure()

    for opt_name, hist in histories.items():
        if not hist.epochs:
            continue
        color = OPTIMIZER_COLORS.get(opt_name, "#FFFFFF")
        
        # Train Loss (Solid line)
        fig.add_trace(go.Scatter(
            x=hist.epochs,
            y=hist.train_losses,
            mode="lines",
            line=dict(color=color, width=2.4),
            name=f"{opt_name} (Train)",
            legendgroup=opt_name
        ))

        # Test Loss (Dashed line)
        fig.add_trace(go.Scatter(
            x=hist.epochs,
            y=hist.test_losses,
            mode="lines",
            line=dict(color=color, width=1.8, dash="dash"),
            name=f"{opt_name} (Test)",
            legendgroup=opt_name
        ))

    fig.update_layout(
        title=dict(
            text="<b>Training & Test Loss vs. Epoch</b>",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=13.5),
            pad=dict(r=70)
        ),
        xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Binary Cross-Entropy Loss", gridcolor="rgba(255,255,255,0.1)"),
        template="plotly_dark",
        margin=dict(l=50, r=25, t=100, b=45),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.88,
            xanchor="center",
            x=0.5,
            font=dict(size=9.0),
            bgcolor="rgba(0,0,0,0)"
        ),
        modebar=dict(
            orientation="h",
            bgcolor="rgba(0,0,0,0)",
            activecolor="#58A6FF"
        ),
        height=430
    )
    return fig


def create_nn_accuracy_figure(histories: Dict[str, TrainingHistory]) -> go.Figure:
    """Create training and test accuracy curves for Part B live dashboard."""
    fig = go.Figure()

    for opt_name, hist in histories.items():
        if not hist.epochs:
            continue
        color = OPTIMIZER_COLORS.get(opt_name, "#FFFFFF")

        fig.add_trace(go.Scatter(
            x=hist.epochs,
            y=[acc * 100 for acc in hist.train_accuracies],
            mode="lines",
            line=dict(color=color, width=2.4),
            name=f"{opt_name} (Train)",
            legendgroup=opt_name
        ))

        fig.add_trace(go.Scatter(
            x=hist.epochs,
            y=[acc * 100 for acc in hist.test_accuracies],
            mode="lines",
            line=dict(color=color, width=1.8, dash="dash"),
            name=f"{opt_name} (Test)",
            legendgroup=opt_name
        ))

    fig.update_layout(
        title=dict(
            text="<b>Classification Accuracy (%) vs. Epoch</b>",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=13.5),
            pad=dict(r=70)
        ),
        xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Accuracy (%)", range=[40, 102], gridcolor="rgba(255,255,255,0.1)"),
        template="plotly_dark",
        margin=dict(l=50, r=25, t=100, b=45),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.88,
            xanchor="center",
            x=0.5,
            font=dict(size=9.0),
            bgcolor="rgba(0,0,0,0)"
        ),
        modebar=dict(
            orientation="h",
            bgcolor="rgba(0,0,0,0)",
            activecolor="#58A6FF"
        ),
        height=430
    )
    return fig


def create_effective_lr_figure(histories: Dict[str, TrainingHistory]) -> go.Figure:
    """
    Live Effective Learning Rate readout for one representative weight (W1[0,0])
    for AdaGrad, RMSProp, Adam, AdamW per PDF Section B2.
    """
    fig = go.Figure()
    adaptive_opts = ["AdaGrad", "RMSProp", "Adam", "AdamW"]
    has_adaptive = False

    for opt_name, hist in histories.items():
        if opt_name in adaptive_opts and hist.epochs:
            has_adaptive = True
            color = OPTIMIZER_COLORS.get(opt_name, "#FFFFFF")

            fig.add_trace(go.Scatter(
                x=hist.epochs,
                y=hist.effective_lrs,
                mode="lines",
                line=dict(color=color, width=2.4),
                name=f"{opt_name} (η_eff)",
                legendgroup=opt_name
            ))

    if not has_adaptive:
        fig.add_annotation(
            text="Effective learning rate is shown for adaptive optimizers<br>(AdaGrad, RMSProp, Adam, AdamW).",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12, color="#8B949E")
        )

    fig.update_layout(
        title=dict(
            text="<b>Effective Learning Rate (η_eff) — Weight W₁[0,0] vs. Epoch</b>",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(size=13),
            pad=dict(r=70)
        ),
        xaxis=dict(title="Epoch", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(
            title="Effective Step Size (η / √(v̂ₜ + ε))",
            type="log" if has_adaptive else "linear",
            gridcolor="rgba(255,255,255,0.1)"
        ),
        template="plotly_dark",
        margin=dict(l=50, r=25, t=100, b=45),
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
        ),
        height=430
    )
    return fig
