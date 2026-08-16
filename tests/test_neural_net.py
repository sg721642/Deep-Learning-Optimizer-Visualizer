"""
Unit tests for BinaryMLP forward pass, loss, accuracy, and numerical gradient checking.
"""
import pytest
import numpy as np
from src.neural_net import BinaryMLP


def test_mlp_shapes():
    mlp = BinaryMLP(input_dim=30, hidden1=16, hidden2=8, output_dim=1, seed=42)
    X = np.random.randn(20, 30)
    y = np.random.randint(0, 2, size=(20, 1)).astype(float)
    
    y_pred, cache = mlp.forward(X)
    assert y_pred.shape == (20, 1)
    assert np.all((y_pred >= 0.0) & (y_pred <= 1.0))
    
    loss = mlp.compute_loss(y, y_pred)
    assert loss >= 0.0

    acc = mlp.compute_accuracy(y, y_pred)
    assert 0.0 <= acc <= 1.0

    grads = mlp.backward(y, cache)
    assert grads["W1"].shape == mlp.params["W1"].shape
    assert grads["b1"].shape == mlp.params["b1"].shape
    assert grads["W2"].shape == mlp.params["W2"].shape
    assert grads["b2"].shape == mlp.params["b2"].shape
    assert grads["W3"].shape == mlp.params["W3"].shape
    assert grads["b3"].shape == mlp.params["b3"].shape


def test_numerical_gradient_check():
    """
    Perform finite-difference numerical gradient checking:
    grad_approx = (L(theta + eps) - L(theta - eps)) / (2 * eps)
    relative_error = ||grad_approx - grad_analytical|| / (||grad_approx|| + ||grad_analytical||)
    Should be < 1e-5.
    """
    np.random.seed(123)
    mlp = BinaryMLP(input_dim=4, hidden1=5, hidden2=3, output_dim=1, seed=123)
    X = np.random.randn(10, 4)
    y = np.random.randint(0, 2, size=(10, 1)).astype(float)

    y_pred, cache = mlp.forward(X)
    analytical_grads = mlp.backward(y, cache)

    eps = 1e-6
    for param_name, param_val in mlp.params.items():
        grad_analytic = analytical_grads[param_name]
        grad_numeric = np.zeros_like(param_val)

        it = np.nditer(param_val, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig_val = param_val[idx]

            # theta + eps
            param_val[idx] = orig_val + eps
            out_plus, _ = mlp.forward(X)
            loss_plus = mlp.compute_loss(y, out_plus)

            # theta - eps
            param_val[idx] = orig_val - eps
            out_minus, _ = mlp.forward(X)
            loss_minus = mlp.compute_loss(y, out_minus)

            # Numerical gradient
            grad_numeric[idx] = (loss_plus - loss_minus) / (2.0 * eps)
            param_val[idx] = orig_val
            it.iternext()

        # Compute relative error
        numerator = np.linalg.norm(grad_analytic - grad_numeric)
        denominator = np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numeric) + 1e-12
        rel_error = numerator / denominator

        assert rel_error < 1e-4, f"Gradient check failed for {param_name} with relative error {rel_error}"
