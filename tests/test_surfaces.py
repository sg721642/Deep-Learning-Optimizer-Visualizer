"""
Unit tests for 2D loss surfaces and simulation engine.
"""
import pytest
import numpy as np
from src.surfaces import SURFACES, get_surface
from src.optimizers import SGD, Adam


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


def test_surface_analytical_gradient():
    s = SURFACES["L2: x² + 50y² (Default)"]
    pt = np.array([3.0, 4.0])
    grad = s.gradient(pt)
    # nabla L = [2*1*3, 2*50*4] = [6, 400]
    np.testing.assert_allclose(grad, [6.0, 400.0])
    assert s.loss_point(pt) == 1.0 * (3.0**2) + 50.0 * (4.0**2)


def test_simulation_trajectory():
    s = SURFACES["L2: x² + 50y² (Default)"]
    opt = Adam(lr=0.1)
    traj, losses = s.simulate_trajectory(opt, start_point=(8.0, 8.0), max_iters=50)
    assert len(traj) == len(losses)
    assert traj.shape[1] == 2
    assert losses[0] == s.loss_point(np.array([8.0, 8.0]))
    # Adam should reduce loss on this convex bowl
    assert losses[-1] < losses[0]
