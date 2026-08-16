"""
From SGD to AdamW — Deep Learning Optimizer Visualizer
2D Loss Surfaces (L1, L2, L3, L4), analytical gradients, Hessians, and conditioning metrics.
"""
from typing import Dict, Tuple, List, Union, Optional
import numpy as np
from src.optimizers import BaseOptimizer, NAG


class LossSurface2D:
    """Represents a 2D mathematical loss surface for optimizer benchmarking."""

    def __init__(self, name: str, formula_str: str, a: float, b: float):
        self.name = name
        self.formula_str = formula_str
        self.a = float(a)  # coefficient for x^2
        self.b = float(b)  # coefficient for y^2
        self.global_minimum = np.array([0.0, 0.0])

    @property
    def condition_number(self) -> float:
        """Hessian condition number kappa = lambda_max / lambda_min."""
        # Hessian H = [[2*a, 0], [0, 2*b]]
        eig1 = 2.0 * self.a
        eig2 = 2.0 * self.b
        return max(eig1, eig2) / min(eig1, eig2)

    def loss(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute L(x, y) = a * x^2 + b * y^2."""
        return self.a * (x ** 2) + self.b * (y ** 2)

    def loss_point(self, point: np.ndarray) -> float:
        """Compute loss at a 1D coordinate array [x, y]."""
        return float(self.a * (point[0] ** 2) + self.b * (point[1] ** 2))

    def gradient(self, point: np.ndarray) -> np.ndarray:
        """Compute analytical gradient nabla L(x, y) = [2 * a * x, 2 * b * y]."""
        return np.array([2.0 * self.a * point[0], 2.0 * self.b * point[1]], dtype=float)

    def hessian(self) -> np.ndarray:
        """Compute analytical Hessian matrix."""
        return np.array([[2.0 * self.a, 0.0], [0.0, 2.0 * self.b]], dtype=float)

    def simulate_trajectory(
        self,
        optimizer: BaseOptimizer,
        start_point: Tuple[float, float] = (8.0, 8.0),
        max_iters: int = 500,
        clip_threshold: float = 1e6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the optimizer from start_point on this surface for max_iters steps.
        Returns:
            trajectory: shape (N+1, 2) array of (x, y) coordinates
            losses: shape (N+1,) array of loss values at each iteration
        """
        optimizer.reset()
        current_pos = np.array(start_point, dtype=float)
        
        trajectory: List[np.ndarray] = [current_pos.copy()]
        losses: List[float] = [self.loss_point(current_pos)]

        for _ in range(max_iters):
            # If optimizer is NAG, evaluate gradient at lookahead position
            if isinstance(optimizer, NAG):
                lookahead_pos = optimizer.get_lookahead_params(current_pos)
                # Check for numerical explosion before grad eval
                if np.isnan(lookahead_pos).any() or np.isinf(lookahead_pos).any() or np.abs(lookahead_pos).max() > clip_threshold:
                    break
                grad = self.gradient(lookahead_pos)
            else:
                grad = self.gradient(current_pos)

            # Check if gradient or position is NaN/inf or exploding
            if np.isnan(grad).any() or np.isinf(grad).any() or np.abs(current_pos).max() > clip_threshold:
                break

            current_pos = optimizer.step(current_pos, grad)
            
            # Record state
            trajectory.append(current_pos.copy())
            loss_val = self.loss_point(current_pos)
            losses.append(loss_val)

            # If exploded beyond threshold, terminate cleanly
            if np.isnan(loss_val) or np.isinf(loss_val) or loss_val > clip_threshold:
                break

        return np.array(trajectory), np.array(losses)


# Registry of 4 surfaces required by PDF
SURFACES: Dict[str, LossSurface2D] = {
    "L1: x² + 10y²": LossSurface2D("L1", "x² + 10y²", a=1.0, b=10.0),
    "L2: x² + 50y² (Default)": LossSurface2D("L2", "x² + 50y²", a=1.0, b=50.0),
    "L3: x² + 100y²": LossSurface2D("L3", "x² + 100y²", a=1.0, b=100.0),
    "L4: x² + 1000y²": LossSurface2D("L4", "x² + 1000y²", a=1.0, b=1000.0),
}

DEFAULT_SURFACE_KEY = "L2: x² + 50y² (Default)"


def get_surface(key: str) -> LossSurface2D:
    """Retrieve surface instance by key."""
    if key not in SURFACES:
        raise ValueError(f"Unknown surface key: {key}. Available: {list(SURFACES.keys())}")
    return SURFACES[key]
