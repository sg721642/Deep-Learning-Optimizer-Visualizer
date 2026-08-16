"""
Unit tests for from-scratch NumPy optimizers.
"""
import pytest
import numpy as np
from src.optimizers import (
    SGD,
    Momentum,
    NAG,
    AdaGrad,
    RMSProp,
    Adam,
    AdamW,
    get_optimizer,
)


def test_sgd_step():
    opt = SGD(lr=0.1)
    params = np.array([2.0, -3.0])
    grads = np.array([1.0, 2.0])
    new_params = opt.step(params, grads)
    # theta_1 = [2 - 0.1*1, -3 - 0.1*2] = [1.9, -3.2]
    np.testing.assert_allclose(new_params, [1.9, -3.2])
    assert opt.t == 1


def test_momentum_step():
    opt = Momentum(lr=0.1, beta=0.9)
    params = np.array([2.0, -3.0])
    grads = np.array([1.0, 2.0])
    
    # Step 1: v_1 = 0.9*0 + 0.1*grads = [0.1, 0.2]
    # theta_1 = [2, -3] - 0.1 * [0.1, 0.2] = [1.99, -3.02]
    new_params = opt.step(params, grads)
    np.testing.assert_allclose(new_params, [1.99, -3.02])
    assert opt.t == 1

    # Step 2: grads = [1, 2]
    # v_2 = 0.9*[0.1, 0.2] + 0.1*[1, 2] = [0.19, 0.38]
    # theta_2 = [1.99, -3.02] - 0.1*[0.19, 0.38] = [1.971, -3.058]
    new_params_2 = opt.step(new_params, grads)
    np.testing.assert_allclose(new_params_2, [1.971, -3.058])


def test_nag_lookahead():
    opt = NAG(lr=0.1, beta=0.9)
    params = np.array([2.0, -3.0])
    # Initially v=0, lookahead should equal params
    la = opt.get_lookahead_params(params)
    np.testing.assert_allclose(la, params)

    # Step 1
    grads = np.array([1.0, 2.0])
    new_params = opt.step(params, grads)
    # Now v is [0.1, 0.2]
    # Lookahead from new_params: new_params - lr * beta * v_prev = new_params - 0.1 * 0.9 * [0.1, 0.2]
    la_2 = opt.get_lookahead_params(new_params)
    expected_la = new_params - 0.1 * 0.9 * np.array([0.1, 0.2])
    np.testing.assert_allclose(la_2, expected_la)


def test_adagrad_step():
    opt = AdaGrad(lr=0.1, eps=1e-8)
    params = np.array([2.0, -3.0])
    grads = np.array([3.0, 4.0])
    # G_1 = [9.0, 16.0]
    # sqrt(G_1) = [3.0, 4.0]
    # eff_lr = 0.1 / [3.0, 4.0]
    # theta_1 = [2.0, -3.0] - 0.1 / [3, 4] * [3, 4] = [2.0 - 0.1, -3.0 - 0.1] = [1.9, -3.1]
    new_params = opt.step(params, grads)
    np.testing.assert_allclose(new_params, [1.9, -3.1], rtol=1e-5)


def test_rmsprop_step():
    opt = RMSProp(lr=0.1, beta=0.9, eps=1e-8)
    params = np.array([2.0, -3.0])
    grads = np.array([2.0, 4.0])
    # v_1 = 0.9*0 + 0.1 * [4, 16] = [0.4, 1.6]
    new_params = opt.step(params, grads)
    expected_delta = 0.1 * grads / np.sqrt(0.1 * (grads ** 2) + 1e-8)
    np.testing.assert_allclose(new_params, params - expected_delta, rtol=1e-5)


def test_adam_step():
    opt = Adam(lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)
    params = np.array([1.0, 2.0])
    grads = np.array([0.5, -0.5])
    new_params = opt.step(params, grads)
    # Manual Adam calculation for step 1:
    m1 = 0.1 * grads
    v1 = 0.001 * (grads ** 2)
    m_hat = m1 / (1.0 - 0.9)  # = grads
    v_hat = v1 / (1.0 - 0.999)  # = grads ** 2
    expected = params - 0.01 * np.sign(grads)
    np.testing.assert_allclose(new_params, expected, rtol=1e-4)


def test_adamw_decoupled_weight_decay():
    lr = 0.01
    wd = 0.05
    opt_adamw = AdamW(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=wd)
    opt_adam = Adam(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)

    params = np.array([10.0, -10.0])
    grads = np.array([0.5, -0.5])

    res_adam = opt_adam.step(params.copy(), grads)
    res_adamw = opt_adamw.step(params.copy(), grads)

    # In AdamW, parameter update contains additional - lr * wd * params
    diff = res_adam - res_adamw
    np.testing.assert_allclose(diff, lr * wd * params, rtol=1e-5)


def test_dict_parameters_support():
    """Verify that optimizers support dictionaries of arrays for neural network weights."""
    opt = Adam(lr=0.01)
    params = {
        "W1": np.ones((16, 30)),
        "b1": np.zeros((16, 1)),
        "W2": np.ones((8, 16)),
        "b2": np.zeros((8, 1))
    }
    grads = {
        "W1": np.full((16, 30), 0.1),
        "b1": np.full((16, 1), 0.1),
        "W2": np.full((8, 16), 0.1),
        "b2": np.full((8, 1), 0.1)
    }
    updated = opt.step(params, grads)
    assert isinstance(updated, dict)
    assert set(updated.keys()) == set(params.keys())
    assert updated["W1"].shape == (16, 30)
    assert updated["b1"].shape == (16, 1)


def test_invalid_hyperparameters():
    with pytest.raises(ValueError):
        SGD(lr=-0.01)
    with pytest.raises(ValueError):
        Momentum(lr=0.01, beta=1.5)
    with pytest.raises(ValueError):
        Adam(lr=0.01, beta1=-0.1)
    with pytest.raises(ValueError):
        AdamW(lr=0.01, weight_decay=-0.1)


def test_effective_learning_rate_computation():
    """Verify effective learning rate formulas for adaptive optimizers."""
    grads = np.array([2.0, 4.0])
    
    adagrad = AdaGrad(lr=0.01, eps=1e-8)
    adagrad.step(np.array([1.0, 1.0]), grads)
    expected_adagrad_eff = 0.01 / np.sqrt(grads[0]**2 + 1e-8)
    np.testing.assert_allclose(adagrad.get_effective_lr(index=(0,)), expected_adagrad_eff, rtol=1e-4)

    rmsprop = RMSProp(lr=0.01, beta=0.9, eps=1e-8)
    rmsprop.step(np.array([1.0, 1.0]), grads)
    expected_rms_eff = 0.01 / np.sqrt(0.1 * (grads[0]**2) + 1e-8)
    np.testing.assert_allclose(rmsprop.get_effective_lr(index=(0,)), expected_rms_eff, rtol=1e-4)
