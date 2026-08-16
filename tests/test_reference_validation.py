"""
Comprehensive Mathematical Correctness and Independent Reference Validation Tests.
"""
import pytest
import numpy as np
from src.surfaces import SURFACES, LossSurface2D
from src.optimizers import (
    SGD,
    Momentum,
    NAG,
    AdaGrad,
    RMSProp,
    Adam,
    AdamW,
)
from src.neural_net import BinaryMLP
from tests.reference_optimizers import (
    ref_l2_loss,
    ref_l2_gradient,
    ref_sgd,
    ref_momentum,
    ref_nag,
    ref_adagrad,
    ref_rmsprop,
    ref_adam,
    ref_adamw,
)


def test_l2_surface_loss_and_gradient():
    """Verify analytical loss and gradient on default bowl at theta0=(8, 8)."""
    s = SURFACES["L2: x² + 50y² (Default)"]
    pt = np.array([8.0, 8.0])
    
    # Mathematical loss: 8^2 + 50*(8^2) = 64 + 3200 = 3264
    assert s.loss_point(pt) == 3264.0
    assert ref_l2_loss(8.0, 8.0) == 3264.0
    
    # Mathematical gradient: [2*x, 100*y] = [16, 800]
    grad = s.gradient(pt)
    np.testing.assert_allclose(grad, np.array([16.0, 800.0]), atol=1e-12)
    np.testing.assert_allclose(ref_l2_gradient(8.0, 8.0), np.array([16.0, 800.0]), atol=1e-12)


def test_all_seven_optimizers_one_step_reference_match():
    """Compare 1-step updates between production and independent reference implementations."""
    s = SURFACES["L2: x² + 50y² (Default)"]
    theta0 = np.array([8.0, 8.0])
    lr = 0.01

    # 1. SGD
    prod_sgd = SGD(lr=lr)
    p_traj, p_loss = s.simulate_trajectory(prod_sgd, (8.0, 8.0), max_iters=1)
    r_traj, r_loss = ref_sgd(theta0, lr, 1)
    np.testing.assert_allclose(p_traj, r_traj, atol=1e-12)
    np.testing.assert_allclose(p_loss, r_loss, atol=1e-12)

    # 2. Momentum
    prod_mom = Momentum(lr=lr, beta=0.9)
    p_traj, p_loss = s.simulate_trajectory(prod_mom, (8.0, 8.0), max_iters=1)
    r_traj, r_loss = ref_momentum(theta0, lr, 0.9, 1)
    np.testing.assert_allclose(p_traj, r_traj, atol=1e-12)
    np.testing.assert_allclose(p_loss, r_loss, atol=1e-12)

    # 3. NAG
    prod_nag = NAG(lr=lr, beta=0.9)
    p_traj, p_loss = s.simulate_trajectory(prod_nag, (8.0, 8.0), max_iters=1)
    r_traj, r_loss = ref_nag(theta0, lr, 0.9, 1)
    np.testing.assert_allclose(p_traj, r_traj, atol=1e-12)
    np.testing.assert_allclose(p_loss, r_loss, atol=1e-12)

    # 4. AdaGrad
    prod_ada = AdaGrad(lr=lr, eps=1e-8)
    p_traj, p_loss = s.simulate_trajectory(prod_ada, (8.0, 8.0), max_iters=1)
    r_traj, r_loss = ref_adagrad(theta0, lr, 1e-8, 1)
    np.testing.assert_allclose(p_traj, r_traj, atol=1e-12)
    np.testing.assert_allclose(p_loss, r_loss, atol=1e-12)

    # 5. RMSProp
    prod_rms = RMSProp(lr=lr, beta=0.9, eps=1e-8)
    p_traj, p_loss = s.simulate_trajectory(prod_rms, (8.0, 8.0), max_iters=1)
    r_traj, r_loss = ref_rmsprop(theta0, lr, 0.9, 1e-8, 1)
    np.testing.assert_allclose(p_traj, r_traj, atol=1e-12)
    np.testing.assert_allclose(p_loss, r_loss, atol=1e-12)

    # 6. Adam
    prod_adam = Adam(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)
    p_traj, p_loss = s.simulate_trajectory(prod_adam, (8.0, 8.0), max_iters=1)
    r_traj, r_loss = ref_adam(theta0, lr, 0.9, 0.999, 1e-8, 1)
    np.testing.assert_allclose(p_traj, r_traj, atol=1e-12)
    np.testing.assert_allclose(p_loss, r_loss, atol=1e-12)

    # 7. AdamW
    prod_adamw = AdamW(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.001)
    p_traj, p_loss = s.simulate_trajectory(prod_adamw, (8.0, 8.0), max_iters=1)
    r_traj, r_loss = ref_adamw(theta0, lr, 0.9, 0.999, 1e-8, 0.001, 1)
    np.testing.assert_allclose(p_traj, r_traj, atol=1e-12)
    np.testing.assert_allclose(p_loss, r_loss, atol=1e-12)


def test_all_seven_optimizers_multi_step_trajectory_match():
    """Verify 500-step numerical identity between production and independent reference code."""
    s = SURFACES["L2: x² + 50y² (Default)"]
    theta0 = np.array([8.0, 8.0])
    lr = 0.01
    steps = 500

    checkpoints = [0, 1, 2, 5, 10, 25, 50, 100, 250, 500]

    experiments = [
        ("SGD", SGD(lr=lr), ref_sgd(theta0, lr, steps)),
        ("Momentum", Momentum(lr=lr, beta=0.9), ref_momentum(theta0, lr, 0.9, steps)),
        ("NAG", NAG(lr=lr, beta=0.9), ref_nag(theta0, lr, 0.9, steps)),
        ("AdaGrad", AdaGrad(lr=lr, eps=1e-8), ref_adagrad(theta0, lr, 1e-8, steps)),
        ("RMSProp", RMSProp(lr=lr, beta=0.9, eps=1e-8), ref_rmsprop(theta0, lr, 0.9, 1e-8, steps)),
        ("Adam", Adam(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8), ref_adam(theta0, lr, 0.9, 0.999, 1e-8, steps)),
        ("AdamW", AdamW(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.001), ref_adamw(theta0, lr, 0.9, 0.999, 1e-8, 0.001, steps)),
    ]

    for name, prod_opt, (r_traj, r_loss) in experiments:
        p_traj, p_loss = s.simulate_trajectory(prod_opt, (8.0, 8.0), max_iters=steps)
        for cp in checkpoints:
            np.testing.assert_allclose(
                p_traj[cp], r_traj[cp], atol=1e-11,
                err_msg=f"{name} trajectory mismatch at step {cp}"
            )
            np.testing.assert_allclose(
                p_loss[cp], r_loss[cp], atol=1e-11,
                err_msg=f"{name} loss mismatch at step {cp}"
            )


def test_trajectory_loss_consistency_across_all_surfaces():
    """Verify that loss_history[t] == L(trajectory[t]) for all 4 surfaces and all 7 optimizers."""
    optimizers = [
        SGD(0.01),
        Momentum(0.01, 0.9),
        NAG(0.01, 0.9),
        AdaGrad(0.01),
        RMSProp(0.01, 0.9),
        Adam(0.01),
        AdamW(0.01, weight_decay=0.001)
    ]

    for s_key, surf in SURFACES.items():
        for opt in optimizers:
            traj, losses = surf.simulate_trajectory(opt, (8.0, 8.0), max_iters=50)
            for t in range(len(traj)):
                computed_loss = surf.loss_point(traj[t])
                assert abs(computed_loss - losses[t]) < 1e-10, (
                    f"Inconsistency on {s_key} with {opt.name} at step {t}: "
                    f"computed {computed_loss} vs recorded {losses[t]}"
                )


def test_analytical_gradients_vs_finite_differences():
    """Verify analytical gradients against central finite differences on all 4 surfaces."""
    h = 1e-6
    np.random.seed(42)

    for s_key, surf in SURFACES.items():
        # Test 10 random points across each landscape
        random_points = np.random.uniform(-15.0, 15.0, size=(10, 2))
        for pt in random_points:
            analytical_grad = surf.gradient(pt)
            
            # Numerical gradient via central differences
            fx_plus = surf.loss_point(pt + np.array([h, 0.0]))
            fx_minus = surf.loss_point(pt - np.array([h, 0.0]))
            num_gx = (fx_plus - fx_minus) / (2.0 * h)

            fy_plus = surf.loss_point(pt + np.array([0.0, h]))
            fy_minus = surf.loss_point(pt - np.array([0.0, h]))
            num_gy = (fy_plus - fy_minus) / (2.0 * h)

            numerical_grad = np.array([num_gx, num_gy])

            np.testing.assert_allclose(
                analytical_grad, numerical_grad, rtol=1e-5, atol=1e-5,
                err_msg=f"Gradient discrepancy on {s_key} at {pt}"
            )


def test_adamw_reduces_to_adam_when_lambda_zero():
    """When weight_decay=0, AdamW must produce trajectories identical to Adam."""
    s = SURFACES["L2: x² + 50y² (Default)"]
    lr = 0.01
    
    adam = Adam(lr=lr, beta1=0.9, beta2=0.999)
    adamw_zero = AdamW(lr=lr, beta1=0.9, beta2=0.999, weight_decay=0.0)

    traj_adam, loss_adam = s.simulate_trajectory(adam, (8.0, 8.0), max_iters=100)
    traj_adamw, loss_adamw = s.simulate_trajectory(adamw_zero, (8.0, 8.0), max_iters=100)

    np.testing.assert_allclose(traj_adam, traj_adamw, atol=1e-14)
    np.testing.assert_allclose(loss_adam, loss_adamw, atol=1e-14)


def test_neural_network_backprop_finite_differences():
    """Verify analytical backpropagation gradients against finite differences for all layers."""
    nn = BinaryMLP(input_dim=4, hidden1=3, hidden2=2, output_dim=1, seed=42)
    X = np.random.randn(5, 4)
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])

    _, cache = nn.forward(X)
    analytical_grads = nn.backward(y, cache)

    h = 1e-6
    for param_name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        param = nn.params[param_name]
        grad_ana = analytical_grads[param_name]
        grad_num = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig_val = param[idx]

            # f(x + h)
            param[idx] = orig_val + h
            pred_plus, _ = nn.forward(X)
            loss_plus = nn.compute_loss(y, pred_plus)

            # f(x - h)
            param[idx] = orig_val - h
            pred_minus, _ = nn.forward(X)
            loss_minus = nn.compute_loss(y, pred_minus)

            param[idx] = orig_val
            grad_num[idx] = (loss_plus - loss_minus) / (2.0 * h)
            it.iternext()

        np.testing.assert_allclose(
            grad_ana, grad_num, rtol=1e-4, atol=1e-4,
            err_msg=f"Backprop gradient discrepancy for parameter {param_name}"
        )
