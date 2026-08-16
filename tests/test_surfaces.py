"""
Unit tests for 2D loss surfaces, analytical gradients, and simulation engine.
"""
import pytest
import numpy as np
from src.surfaces import SURFACES, get_surface
from src.optimizers import (
    SGD,
    Momentum,
    NAG,
    AdaGrad,
    RMSProp,
    Adam,
    AdamW,
)


def test_surfaces_registry():
    assert "L1: x² + 10y²" in SURFACES
    assert "L2: x² + 50y² (Default)" in SURFACES
    assert "L3: x² + 100y²" in SURFACES
    assert "L4: x² + 1000y²" in SURFACES


def test_condition_numbers():
    assert SURFACES["L1: x² + 10y²"].condition_number == 10.0
    assert SURFACES["L2: x² + 50y² (Default)"].condition_number == 50.0
    assert SURFACES["L3: x² + 100y²"].condition_number == 100.0
    assert SURFACES["L4: x² + 1000y²"].condition_number == 1000.0


def test_surface_l2_gradient_at_start():
    """Verify gradient of L2 = x^2 + 50y^2 at start point (8, 8) is [16, 800]."""
    s = SURFACES["L2: x² + 50y² (Default)"]
    start_pt = np.array([8.0, 8.0])
    
    # Loss at (8, 8): 8^2 + 50 * 8^2 = 64 + 3200 = 3264
    assert s.loss_point(start_pt) == 3264.0
    
    # Gradient at (8, 8): [2*x, 100*y] = [16, 800]
    grad = s.gradient(start_pt)
    np.testing.assert_allclose(grad, [16.0, 800.0])


def test_all_seven_optimizers_one_step_on_l2():
    """Verify 1-step parameter updates for all 7 optimizers on L2 with lr=0.01."""
    s = SURFACES["L2: x² + 50y² (Default)"]
    start_pt = (8.0, 8.0)
    lr = 0.01

    opts = {
        "SGD": SGD(lr=lr),
        "Momentum": Momentum(lr=lr, beta=0.9),
        "NAG": NAG(lr=lr, beta=0.9),
        "AdaGrad": AdaGrad(lr=lr),
        "RMSProp": RMSProp(lr=lr, beta=0.9),
        "Adam": Adam(lr=lr, beta1=0.9, beta2=0.999),
        "AdamW": AdamW(lr=lr, beta1=0.9, beta2=0.999, weight_decay=0.001),
    }

    expected_step1 = {
        "SGD": np.array([7.84, 0.0]),
        "Momentum": np.array([7.984, 7.200]),
        "NAG": np.array([7.984, 7.200]),
        "AdaGrad": np.array([7.990, 7.990]),
        "RMSProp": np.array([7.968377, 7.968377]),
        "Adam": np.array([7.990, 7.990]),
        "AdamW": np.array([7.98992, 7.98992]),
    }

    for name, opt in opts.items():
        traj, losses = s.simulate_trajectory(opt, start_point=start_pt, max_iters=1)
        assert len(traj) == 2
        np.testing.assert_allclose(traj[1], expected_step1[name], rtol=1e-4, err_msg=f"1-step mismatch for {name}")
        assert losses[1] < losses[0]


def test_multi_step_trajectory_convergence():
    """Verify that optimizers reduce loss over 100 iterations on L2."""
    s = SURFACES["L2: x² + 50y² (Default)"]
    start_pt = (8.0, 8.0)
    
    for opt in [SGD(0.01), Momentum(0.01, 0.9), NAG(0.01, 0.9), Adam(0.01)]:
        traj, losses = s.simulate_trajectory(opt, start_point=start_pt, max_iters=100)
        assert losses[-1] < losses[0]
        if opt.name in ["SGD", "Momentum", "NAG"]:
            assert losses[-1] < 5.0
