"""
From SGD to AdamW — Deep Learning Optimizer Visualizer
Modular from-scratch implementations of 7 optimizers in pure NumPy.
"""
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np

# Constant color map for visual consistency across all views in the application
OPTIMIZER_COLORS: Dict[str, str] = {
    "SGD": "#E63946",            # Vibrant Red
    "Momentum": "#F4A261",       # Warm Orange
    "NAG": "#E76F51",            # Coral Red
    "AdaGrad": "#2A9D8F",        # Teal / Green
    "RMSProp": "#457B9D",        # Steel Blue
    "Adam": "#1D3557",           # Deep Navy Blue
    "AdamW": "#9B5DE5",          # Royal Purple
}

OPTIMIZER_DESCRIPTIONS: Dict[str, str] = {
    "SGD": "Standard Stochastic Gradient Descent. Directly updates parameters in the direction of the negative gradient.",
    "Momentum": "SGD with Momentum. Accumulates past gradients as velocity to dampen oscillations and accelerate in consistent directions.",
    "NAG": "Nesterov Accelerated Gradient. Evaluates gradients at a look-ahead position (θ - η·β·v) to apply anticipatory braking.",
    "AdaGrad": "Adaptive Gradient Algorithm. Scales learning rates inversely proportional to the cumulative sum of squared gradients.",
    "RMSProp": "Root Mean Square Propagation. Resolves AdaGrad's diminishing learning rate using an exponential moving average of squared gradients.",
    "Adam": "Adaptive Moment Estimation. Combines first moments (momentum) and second moments (RMSProp) with bias correction.",
    "AdamW": "Adam with Decoupled Weight Decay. Applies weight decay directly to parameters, preserving true regularization independent of adaptive gradient scales."
}


class BaseOptimizer:
    """Base class for all 7 from-scratch NumPy optimizers."""

    def __init__(self, lr: float = 0.01, name: str = "BaseOptimizer"):
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.lr = float(lr)
        self.name = name
        self.t = 0
        self.last_effective_lr: Dict[str, np.ndarray] = {}

    def reset(self):
        """Reset internal optimizer state and timestep."""
        self.t = 0
        self.last_effective_lr = {}

    def get_lookahead_params(self, params: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Return parameters for look-ahead gradient evaluation (used by NAG)."""
        if isinstance(params, dict):
            return {k: v.copy() for k, v in params.items()}
        return np.copy(params)

    def step(
        self,
        params: Union[np.ndarray, Dict[str, np.ndarray]],
        grads: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Perform a single optimization step."""
        raise NotImplementedError

    def get_effective_lr(
        self,
        key: Optional[str] = None,
        index: Optional[Tuple[int, ...]] = None
    ) -> float:
        """
        Return the effective learning rate for a specific parameter coordinate.
        Defaults to the overall learning rate if no adaptive scaling applies.
        """
        if not self.last_effective_lr:
            return self.lr
        if key is not None and key in self.last_effective_lr:
            arr = self.last_effective_lr[key]
        else:
            arr = next(iter(self.last_effective_lr.values()))
        
        if index is not None and isinstance(arr, np.ndarray) and arr.ndim > 0:
            try:
                return float(arr[index])
            except Exception:
                return float(arr.flatten()[0])
        elif isinstance(arr, np.ndarray):
            return float(arr.flatten()[0]) if arr.size > 0 else self.lr
        return float(arr)


class SGD(BaseOptimizer):
    """
    1. Stochastic Gradient Descent (SGD)
    Update Rule:
        θ_{t+1} = θ_t - η * g_t
    """
    def __init__(self, lr: float = 0.01):
        super().__init__(lr=lr, name="SGD")

    def step(
        self,
        params: Union[np.ndarray, Dict[str, np.ndarray]],
        grads: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        self.t += 1
        is_dict = isinstance(params, dict)
        p_dict = params if is_dict else {"param": params}
        g_dict = grads if is_dict else {"param": grads}

        updated = {}
        for k in p_dict:
            p = p_dict[k]
            g = g_dict[k]
            updated[k] = p - self.lr * g
            self.last_effective_lr[k] = np.full_like(p, self.lr)

        return updated if is_dict else updated["param"]


class Momentum(BaseOptimizer):
    """
    2. SGD with Momentum
    Update Rule (per PDF spec):
        v_t = β * v_{t-1} + (1 - β) * g_t
        θ_{t+1} = θ_t - η * v_t
    """
    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        super().__init__(lr=lr, name="Momentum")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Beta must be in [0, 1), got {beta}")
        self.beta = float(beta)
        self.v: Dict[str, np.ndarray] = {}

    def reset(self):
        super().reset()
        self.v = {}

    def step(
        self,
        params: Union[np.ndarray, Dict[str, np.ndarray]],
        grads: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        self.t += 1
        is_dict = isinstance(params, dict)
        p_dict = params if is_dict else {"param": params}
        g_dict = grads if is_dict else {"param": grads}

        updated = {}
        for k in p_dict:
            p = p_dict[k]
            g = g_dict[k]
            if k not in self.v:
                self.v[k] = np.zeros_like(p)

            # v_t = β * v_{t-1} + (1 - β) * g_t
            self.v[k] = self.beta * self.v[k] + (1.0 - self.beta) * g
            # θ_{t+1} = θ_t - η * v_t
            updated[k] = p - self.lr * self.v[k]
            self.last_effective_lr[k] = np.full_like(p, self.lr)

        return updated if is_dict else updated["param"]


class NAG(BaseOptimizer):
    """
    3. Nesterov Accelerated Gradient (NAG)
    Update Rule (per PDF spec & dimensional analysis):
        θ_lookahead = θ_t - η * β * v_{t-1}
        v_t = β * v_{t-1} + (1 - β) * ∇L(θ_lookahead)
        θ_{t+1} = θ_t - η * v_t

    Note on PDF notation vs Dimensional Consistency:
    The PDF lists the lookahead as `θ_t - β * v_{t-1}` while defining velocity as an EMA
    of gradients `v_t = β * v_{t-1} + (1-β) * g_t` and updating `θ_{t+1} = θ_t - η * v_t`.
    Because `v` has physical dimensions of a gradient ([L]/[θ]) while θ has dimensions of
    parameter coordinates ([θ]), stepping in parameter space requires the step-size factor `η`:
    `θ_lookahead = θ_t - η * β * v_{t-1}`. Without `η`, subtracting gradient-scale quantities
    directly from coordinates creates a 100x unscaled step that causes immediate divergence.
    """
    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        super().__init__(lr=lr, name="NAG")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Beta must be in [0, 1), got {beta}")
        self.beta = float(beta)
        self.v: Dict[str, np.ndarray] = {}

    def reset(self):
        super().reset()
        self.v = {}

    def get_lookahead_params(self, params: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Lookahead position in parameter space: θ_lookahead = θ_t - η * β * v_{t-1}."""
        is_dict = isinstance(params, dict)
        p_dict = params if is_dict else {"param": params}
        lookahead = {}
        for k, p in p_dict.items():
            v_prev = self.v.get(k, np.zeros_like(p))
            lookahead[k] = p - self.lr * self.beta * v_prev
        return lookahead if is_dict else lookahead["param"]

    def step(
        self,
        params: Union[np.ndarray, Dict[str, np.ndarray]],
        grads: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        self.t += 1
        is_dict = isinstance(params, dict)
        p_dict = params if is_dict else {"param": params}
        g_dict = grads if is_dict else {"param": grads}

        updated = {}
        for k in p_dict:
            p = p_dict[k]
            g = g_dict[k]
            if k not in self.v:
                self.v[k] = np.zeros_like(p)

            # Update velocity using look-ahead gradient
            self.v[k] = self.beta * self.v[k] + (1.0 - self.beta) * g
            # Parameter update
            updated[k] = p - self.lr * self.v[k]
            self.last_effective_lr[k] = np.full_like(p, self.lr)

        return updated if is_dict else updated["param"]


class AdaGrad(BaseOptimizer):
    """
    4. AdaGrad (Adaptive Gradient Algorithm)
    Update Rule (per PDF spec):
        G_t = G_{t-1} + g_t^2
        θ_{t+1} = θ_t - (η / √(G_t + ε)) ⊙ g_t
    """
    def __init__(self, lr: float = 0.01, eps: float = 1e-8):
        super().__init__(lr=lr, name="AdaGrad")
        if eps <= 0:
            raise ValueError(f"Epsilon must be positive, got {eps}")
        self.eps = float(eps)
        self.G: Dict[str, np.ndarray] = {}

    def reset(self):
        super().reset()
        self.G = {}

    def step(
        self,
        params: Union[np.ndarray, Dict[str, np.ndarray]],
        grads: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        self.t += 1
        is_dict = isinstance(params, dict)
        p_dict = params if is_dict else {"param": params}
        g_dict = grads if is_dict else {"param": grads}

        updated = {}
        for k in p_dict:
            p = p_dict[k]
            g = g_dict[k]
            if k not in self.G:
                self.G[k] = np.zeros_like(p)

            # G_t = G_{t-1} + g_t^2
            self.G[k] += g ** 2
            # Effective learning rate: η / √(G_t + ε)
            effective_lr = self.lr / np.sqrt(self.G[k] + self.eps)
            self.last_effective_lr[k] = effective_lr
            # θ_{t+1} = θ_t - effective_lr * g_t
            updated[k] = p - effective_lr * g

        return updated if is_dict else updated["param"]


class RMSProp(BaseOptimizer):
    """
    5. RMSProp (Root Mean Square Propagation)
    Update Rule (per PDF spec):
        v_t = β * v_{t-1} + (1 - β) * g_t^2
        θ_{t+1} = θ_t - (η / √(v_t + ε)) ⊙ g_t
    """
    def __init__(self, lr: float = 0.01, beta: float = 0.9, eps: float = 1e-8):
        super().__init__(lr=lr, name="RMSProp")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Beta must be in [0, 1), got {beta}")
        if eps <= 0:
            raise ValueError(f"Epsilon must be positive, got {eps}")
        self.beta = float(beta)
        self.eps = float(eps)
        self.v: Dict[str, np.ndarray] = {}

    def reset(self):
        super().reset()
        self.v = {}

    def step(
        self,
        params: Union[np.ndarray, Dict[str, np.ndarray]],
        grads: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        self.t += 1
        is_dict = isinstance(params, dict)
        p_dict = params if is_dict else {"param": params}
        g_dict = grads if is_dict else {"param": grads}

        updated = {}
        for k in p_dict:
            p = p_dict[k]
            g = g_dict[k]
            if k not in self.v:
                self.v[k] = np.zeros_like(p)

            # v_t = β * v_{t-1} + (1 - β) * g_t^2
            self.v[k] = self.beta * self.v[k] + (1.0 - self.beta) * (g ** 2)
            # Effective learning rate: η / √(v_t + ε)
            effective_lr = self.lr / np.sqrt(self.v[k] + self.eps)
            self.last_effective_lr[k] = effective_lr
            # θ_{t+1} = θ_t - effective_lr * g_t
            updated[k] = p - effective_lr * g

        return updated if is_dict else updated["param"]


class Adam(BaseOptimizer):
    """
    6. Adam (Adaptive Moment Estimation)
    Update Rule (per PDF spec):
        m_t = β_1 * m_{t-1} + (1 - β_1) * g_t
        v_t = β_2 * v_{t-1} + (1 - β_2) * g_t^2
        m̂_t = m_t / (1 - β_1^t)
        v̂_t = v_t / (1 - β_2^t)
        θ_{t+1} = θ_t - η * m̂_t / (√v̂_t + ε)     [ε is OUTSIDE the sqrt per PDF spec]
    """
    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ):
        super().__init__(lr=lr, name="Adam")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Beta2 must be in [0, 1), got {beta2}")
        if eps <= 0:
            raise ValueError(f"Epsilon must be positive, got {eps}")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def reset(self):
        super().reset()
        self.m = {}
        self.v = {}

    def step(
        self,
        params: Union[np.ndarray, Dict[str, np.ndarray]],
        grads: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        self.t += 1
        is_dict = isinstance(params, dict)
        p_dict = params if is_dict else {"param": params}
        g_dict = grads if is_dict else {"param": grads}

        updated = {}
        for k in p_dict:
            p = p_dict[k]
            g = g_dict[k]
            if k not in self.m:
                self.m[k] = np.zeros_like(p)
                self.v[k] = np.zeros_like(p)

            # m_t = β_1 * m_{t-1} + (1 - β_1) * g_t
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            # v_t = β_2 * v_{t-1} + (1 - β_2) * g_t^2
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[k] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1.0 - self.beta2 ** self.t)

            # Effective learning rate: η / (√v̂_t + ε)  — ε is outside the sqrt per PDF spec
            effective_lr = self.lr / (np.sqrt(v_hat) + self.eps)
            self.last_effective_lr[k] = effective_lr

            # θ_{t+1} = θ_t - effective_lr * m̂_t
            updated[k] = p - effective_lr * m_hat

        return updated if is_dict else updated["param"]


class AdamW(BaseOptimizer):
    """
    7. AdamW (Adam with Decoupled Weight Decay)
    Update Rule (per PDF spec):
        m_t = β_1 * m_{t-1} + (1 - β_1) * g_t
        v_t = β_2 * v_{t-1} + (1 - β_2) * g_t^2
        m̂_t = m_t / (1 - β_1^t)
        v̂_t = v_t / (1 - β_2^t)
        θ_{t+1} = θ_t - η * ( m̂_t / (√v̂_t + ε) + λ * θ_t )  [ε is OUTSIDE the sqrt per PDF spec]
    """
    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-3
    ):
        super().__init__(lr=lr, name="AdamW")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Beta2 must be in [0, 1), got {beta2}")
        if eps <= 0:
            raise ValueError(f"Epsilon must be positive, got {eps}")
        if weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def reset(self):
        super().reset()
        self.m = {}
        self.v = {}

    def step(
        self,
        params: Union[np.ndarray, Dict[str, np.ndarray]],
        grads: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        self.t += 1
        is_dict = isinstance(params, dict)
        p_dict = params if is_dict else {"param": params}
        g_dict = grads if is_dict else {"param": grads}

        updated = {}
        for k in p_dict:
            p = p_dict[k]
            g = g_dict[k]
            if k not in self.m:
                self.m[k] = np.zeros_like(p)
                self.v[k] = np.zeros_like(p)

            # m_t = β_1 * m_{t-1} + (1 - β_1) * g_t
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            # v_t = β_2 * v_{t-1} + (1 - β_2) * g_t^2
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[k] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1.0 - self.beta2 ** self.t)

            # Effective adaptive step scale: η / (√v̂_t + ε)  — ε is outside the sqrt per PDF spec
            effective_lr = self.lr / (np.sqrt(v_hat) + self.eps)
            self.last_effective_lr[k] = effective_lr

            # Decoupled weight decay: θ_{t+1} = θ_t * (1 - η * λ) - effective_lr * m̂_t
            updated[k] = p * (1.0 - self.lr * self.weight_decay) - effective_lr * m_hat

        return updated if is_dict else updated["param"]


def get_optimizer(name: str, **kwargs) -> BaseOptimizer:
    """Factory helper to instantiate any of the 7 optimizers by name."""
    optimizers = {
        "SGD": SGD,
        "Momentum": Momentum,
        "NAG": NAG,
        "AdaGrad": AdaGrad,
        "RMSProp": RMSProp,
        "Adam": Adam,
        "AdamW": AdamW,
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Choose from {list(optimizers.keys())}")
    
    cls = optimizers[name]
    # Filter kwargs valid for each constructor
    valid_args = cls.__init__.__code__.co_varnames
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
    return cls(**filtered_kwargs)
