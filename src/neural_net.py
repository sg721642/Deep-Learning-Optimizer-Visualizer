"""
From SGD to AdamW — Deep Learning Optimizer Visualizer
From-scratch NumPy implementation of a Multi-Layer Perceptron (MLP)
Architecture: Input (30) -> Dense(16) -> ReLU -> Dense(8) -> ReLU -> Dense(1) -> Sigmoid
"""
from typing import Dict, Tuple, Any, Optional
import numpy as np


class BinaryMLP:
    """
    Handcrafted 3-Layer Neural Network using only NumPy:
        Layer 1: Dense(in=30, out=16) + ReLU
        Layer 2: Dense(in=16, out=8) + ReLU
        Layer 3: Dense(in=8, out=1) + Sigmoid
    """

    def __init__(self, input_dim: int = 30, hidden1: int = 16, hidden2: int = 8, output_dim: int = 1, seed: int = 42):
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_dim = output_dim
        self.seed = seed
        self.params: Dict[str, np.ndarray] = {}
        self.init_weights(seed=self.seed)

    def init_weights(self, seed: Optional[int] = None):
        """Initialize weights using He (Kaiming) normal for ReLU and Xavier for Sigmoid."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState(self.seed)

        # He initialization for ReLU layers: std = sqrt(2 / n_in)
        self.params["W1"] = rng.randn(self.input_dim, self.hidden1) * np.sqrt(2.0 / self.input_dim)
        self.params["b1"] = np.zeros((1, self.hidden1), dtype=np.float64)

        self.params["W2"] = rng.randn(self.hidden1, self.hidden2) * np.sqrt(2.0 / self.hidden1)
        self.params["b2"] = np.zeros((1, self.hidden2), dtype=np.float64)

        # Xavier / Glorot initialization for Sigmoid output layer: std = sqrt(2 / (n_in + n_out))
        self.params["W3"] = rng.randn(self.hidden2, self.output_dim) * np.sqrt(2.0 / (self.hidden2 + self.output_dim))
        self.params["b3"] = np.zeros((1, self.output_dim), dtype=np.float64)

    def clone_params(self) -> Dict[str, np.ndarray]:
        """Return a deep copy of current parameters."""
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params: Dict[str, np.ndarray]):
        """Set network parameters."""
        self.params = {k: v.copy() for k, v in params.items()}

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def relu_backward(dA: np.ndarray, z: np.ndarray) -> np.ndarray:
        dZ = dA.copy()
        dZ[z <= 0.0] = 0.0
        return dZ

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def forward(
        self,
        X: np.ndarray,
        params: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward propagation.
        Returns:
            A3: Final output probabilities (N, 1)
            cache: Intermediate activations and pre-activations for backpropagation
        """
        p = params if params is not None else self.params

        # Layer 1: Linear + ReLU
        Z1 = np.dot(X, p["W1"]) + p["b1"]
        A1 = self.relu(Z1)

        # Layer 2: Linear + ReLU
        Z2 = np.dot(A1, p["W2"]) + p["b2"]
        A2 = self.relu(Z2)

        # Layer 3: Linear + Sigmoid
        Z3 = np.dot(A2, p["W3"]) + p["b3"]
        A3 = self.sigmoid(Z3)

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2,
            "Z3": Z3,
            "A3": A3
        }
        return A3, cache

    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
        """
        Binary Cross-Entropy Loss:
        L = - 1/N sum [ y * ln(y_pred + eps) + (1 - y) * ln(1 - y_pred + eps) ]
        """
        y_t = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
        y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
        N = y_t.shape[0]
        y_clipped = np.clip(y_p, eps, 1.0 - eps)
        bce = - (1.0 / N) * np.sum(y_t * np.log(y_clipped) + (1.0 - y_t) * np.log(1.0 - y_clipped))
        return float(bce)

    @staticmethod
    def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary classification accuracy."""
        y_t = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
        y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)
        preds = (y_p >= 0.5).astype(np.float64)
        return float(np.mean(preds == y_t))

    def backward(
        self,
        y_true: np.ndarray,
        cache: Dict[str, np.ndarray],
        params: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Full analytical backpropagation.
        Computes exact gradients dL/dW and dL/db for all 3 layers.
        """
        p = params if params is not None else self.params
        y_t = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
        N = float(y_t.shape[0])

        X = cache["X"]
        Z1, A1 = cache["Z1"], cache["A1"]
        Z2, A2 = cache["Z2"], cache["A2"]
        A3 = cache["A3"]

        # Output error derivative for BCE with Sigmoid: dL/dZ3 = (A3 - Y) / N
        dZ3 = (A3 - y_t) / N
        dW3 = np.dot(A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        # Backprop into Layer 2
        dA2 = np.dot(dZ3, p["W3"].T)
        dZ2 = self.relu_backward(dA2, Z2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Backprop into Layer 1
        dA1 = np.dot(dZ2, p["W2"].T)
        dZ1 = self.relu_backward(dA1, Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3
        }
