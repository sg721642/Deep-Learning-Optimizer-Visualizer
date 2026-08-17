"""
Independent Reference Implementations for mathematical validation.
This module does NOT import or rely on any classes in src.optimizers.
It implements the pure mathematical formulas from first principles.
"""
from typing import Tuple, List, Dict
import numpy as np


def ref_l2_loss(x: float, y: float) -> float:
    return float(x**2 + 50.0 * (y**2))


def ref_l2_gradient(x: float, y: float) -> np.ndarray:
    return np.array([2.0 * x, 100.0 * y], dtype=float)


def ref_sgd(theta0: np.ndarray, lr: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pure mathematical SGD: θ_{t+1} = θ_t - η g_t."""
    traj = [theta0.copy()]
    losses = [ref_l2_loss(theta0[0], theta0[1])]
    theta = theta0.copy()
    for _ in range(steps):
        g = ref_l2_gradient(theta[0], theta[1])
        theta = theta - lr * g
        traj.append(theta.copy())
        losses.append(ref_l2_loss(theta[0], theta[1]))
    return np.array(traj), np.array(losses)


def ref_momentum(theta0: np.ndarray, lr: float, beta: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pure mathematical Momentum: v_t = β v_{t-1} + (1-β) g_t, θ_{t+1} = θ_t - η v_t."""
    traj = [theta0.copy()]
    losses = [ref_l2_loss(theta0[0], theta0[1])]
    theta = theta0.copy()
    v = np.zeros_like(theta)
    for _ in range(steps):
        g = ref_l2_gradient(theta[0], theta[1])
        v = beta * v + (1.0 - beta) * g
        theta = theta - lr * v
        traj.append(theta.copy())
        losses.append(ref_l2_loss(theta[0], theta[1]))
    return np.array(traj), np.array(losses)


def ref_nag(theta0: np.ndarray, lr: float, beta: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure mathematical NAG (dimensionally consistent lookahead in parameter space):
        θ_lookahead = θ_t - η * β * v_{t-1}
        v_t = β * v_{t-1} + (1-β) * ∇L(θ_lookahead)
        θ_{t+1} = θ_t - η * v_t
    """
    traj = [theta0.copy()]
    losses = [ref_l2_loss(theta0[0], theta0[1])]
    theta = theta0.copy()
    v = np.zeros_like(theta)
    for _ in range(steps):
        lookahead = theta - lr * beta * v
        g = ref_l2_gradient(lookahead[0], lookahead[1])
        v = beta * v + (1.0 - beta) * g
        theta = theta - lr * v
        traj.append(theta.copy())
        losses.append(ref_l2_loss(theta[0], theta[1]))
    return np.array(traj), np.array(losses)


def ref_adagrad(theta0: np.ndarray, lr: float, eps: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure mathematical AdaGrad:
        G_t = G_{t-1} + g_t^2
        θ_{t+1} = θ_t - (η / √(G_t + ε)) ⊙ g_t
    """
    traj = [theta0.copy()]
    losses = [ref_l2_loss(theta0[0], theta0[1])]
    theta = theta0.copy()
    G = np.zeros_like(theta)
    for _ in range(steps):
        g = ref_l2_gradient(theta[0], theta[1])
        G = G + g**2
        eff_lr = lr / np.sqrt(G + eps)
        theta = theta - eff_lr * g
        traj.append(theta.copy())
        losses.append(ref_l2_loss(theta[0], theta[1]))
    return np.array(traj), np.array(losses)


def ref_rmsprop(theta0: np.ndarray, lr: float, beta: float, eps: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure mathematical RMSProp:
        v_t = β v_{t-1} + (1-β) g_t^2
        θ_{t+1} = θ_t - (η / √(v_t + ε)) ⊙ g_t
    """
    traj = [theta0.copy()]
    losses = [ref_l2_loss(theta0[0], theta0[1])]
    theta = theta0.copy()
    v = np.zeros_like(theta)
    for _ in range(steps):
        g = ref_l2_gradient(theta[0], theta[1])
        v = beta * v + (1.0 - beta) * (g**2)
        eff_lr = lr / np.sqrt(v + eps)
        theta = theta - eff_lr * g
        traj.append(theta.copy())
        losses.append(ref_l2_loss(theta[0], theta[1]))
    return np.array(traj), np.array(losses)


def ref_adam(theta0: np.ndarray, lr: float, beta1: float, beta2: float, eps: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure mathematical Adam (per PDF spec):
        m_t = β1 m_{t-1} + (1-β1) g_t
        v_t = β2 v_{t-1} + (1-β2) g_t^2
        m̂_t = m_t / (1 - β1^t)
        v̂_t = v_t / (1 - β2^t)
        θ_{t+1} = θ_t - η * m̂_t / (√v̂_t + ε)    [ε is OUTSIDE the sqrt per PDF spec]
    """
    traj = [theta0.copy()]
    losses = [ref_l2_loss(theta0[0], theta0[1])]
    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for t in range(1, steps + 1):
        g = ref_l2_gradient(theta[0], theta[1])
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g**2)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        eff_lr = lr / (np.sqrt(v_hat) + eps)
        theta = theta - eff_lr * m_hat
        traj.append(theta.copy())
        losses.append(ref_l2_loss(theta[0], theta[1]))
    return np.array(traj), np.array(losses)


def ref_adamw(theta0: np.ndarray, lr: float, beta1: float, beta2: float, eps: float, weight_decay: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure mathematical AdamW — Decoupled Weight Decay (per PDF spec):
        θ_{t+1} = θ_t * (1 - η * λ) - η * m̂_t / (√v̂_t + ε)    [ε is OUTSIDE the sqrt per PDF spec]
    """
    traj = [theta0.copy()]
    losses = [ref_l2_loss(theta0[0], theta0[1])]
    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for t in range(1, steps + 1):
        g = ref_l2_gradient(theta[0], theta[1])
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g**2)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        eff_lr = lr / (np.sqrt(v_hat) + eps)
        theta = theta * (1.0 - lr * weight_decay) - eff_lr * m_hat
        traj.append(theta.copy())
        losses.append(ref_l2_loss(theta[0], theta[1]))
    return np.array(traj), np.array(losses)
