"""
From SGD to AdamW — Deep Learning Optimizer Visualizer
Neural Network Experiment runner, live metrics tracking, effective learning rate recording,
and automatic comparison table computation.
"""
from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np
import pandas as pd
from src.optimizers import (
    BaseOptimizer,
    NAG,
    AdaGrad,
    RMSProp,
    Adam,
    AdamW,
    get_optimizer,
    OPTIMIZER_COLORS
)
from src.neural_net import BinaryMLP
from src.dataset import DatasetManager


class TrainingHistory:
    """Stores per-epoch training metrics and effective learning rates for an optimizer."""

    def __init__(self, optimizer_name: str):
        self.optimizer_name = optimizer_name
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.test_accuracies: List[float] = []
        self.effective_lrs: List[float] = []

    def record(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float,
        train_acc: float,
        test_acc: float,
        effective_lr: float
    ):
        self.epochs.append(epoch)
        self.train_losses.append(float(train_loss))
        self.test_losses.append(float(test_loss))
        self.train_accuracies.append(float(train_acc))
        self.test_accuracies.append(float(test_acc))
        self.effective_lrs.append(float(effective_lr))

    def compute_convergence_epoch(self) -> str:
        """
        Compute convergence epoch per PDF specification:
        'the first epoch at which validation loss reaches within 1% of its final value'
        Condition: |val_loss(e) - final_val_loss| <= 0.01 * final_val_loss
        """
        if not self.test_losses:
            return "N/A"
        
        final_val_loss = self.test_losses[-1]
        if np.isnan(final_val_loss) or np.isinf(final_val_loss):
            return "Diverged"
        
        threshold = 0.01 * abs(final_val_loss)
        for i, val_loss in enumerate(self.test_losses):
            if not np.isnan(val_loss) and not np.isinf(val_loss):
                if abs(val_loss - final_val_loss) <= threshold:
                    return str(self.epochs[i])
        
        return str(self.epochs[-1])


class NNTrainingEngine:
    """Manages multi-optimizer neural network training with live epoch callbacks."""

    def __init__(self, dataset_manager: DatasetManager):
        self.dm = dataset_manager
        self.X_train, self.y_train, self.X_test, self.y_test = self.dm.get_data()

    def train_single_optimizer(
        self,
        optimizer: BaseOptimizer,
        epochs: int = 100,
        batch_size: int = 32,
        initial_seed: int = 42,
        epoch_callback: Optional[Callable[[int, TrainingHistory], None]] = None
    ) -> TrainingHistory:
        """
        Train a BinaryMLP on the dataset using a specific optimizer.
        Uses identical initial weights for fair cross-optimizer benchmarking.
        """
        # Create fresh MLP with deterministic initial weights
        mlp = BinaryMLP(
            input_dim=self.dm.num_features,
            hidden1=16,
            hidden2=8,
            output_dim=1,
            seed=initial_seed
        )
        optimizer.reset()
        history = TrainingHistory(optimizer_name=optimizer.name)

        num_samples = self.X_train.shape[0]
        # Handle batch size >= num_samples (full batch) or mini-batch
        effective_batch_size = min(batch_size, num_samples) if batch_size > 0 else num_samples

        for epoch in range(1, epochs + 1):
            # Shuffle training data each epoch
            indices = np.random.RandomState(epoch).permutation(num_samples)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]

            # Mini-batch loop
            for start_idx in range(0, num_samples, effective_batch_size):
                end_idx = min(start_idx + effective_batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                if isinstance(optimizer, NAG):
                    # For NAG: evaluate lookahead parameters first
                    lookahead_params = optimizer.get_lookahead_params(mlp.params)
                    _, cache = mlp.forward(X_batch, params=lookahead_params)
                    grads = mlp.backward(y_batch, cache, params=lookahead_params)
                else:
                    _, cache = mlp.forward(X_batch)
                    grads = mlp.backward(y_batch, cache)

                # Check for numerical explosion
                has_nan = any(np.isnan(g).any() or np.isinf(g).any() for g in grads.values())
                if has_nan:
                    break

                # Optimizer step updates MLP parameters
                mlp.params = optimizer.step(mlp.params, grads)

            # Evaluate epoch-level metrics on full train and test sets
            y_pred_train, _ = mlp.forward(self.X_train)
            train_loss = mlp.compute_loss(self.y_train, y_pred_train)
            train_acc = mlp.compute_accuracy(self.y_train, y_pred_train)

            y_pred_test, _ = mlp.forward(self.X_test)
            test_loss = mlp.compute_loss(self.y_test, y_pred_test)
            test_acc = mlp.compute_accuracy(self.y_test, y_pred_test)

            # Extract effective learning rate for representative weight W1[0, 0]
            eff_lr = optimizer.get_effective_lr(key="W1", index=(0, 0))

            history.record(
                epoch=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                train_acc=train_acc,
                test_acc=test_acc,
                effective_lr=eff_lr
            )

            if epoch_callback is not None:
                epoch_callback(epoch, history)

        return history

    def run_all_optimizers(
        self,
        optimizer_configs: Dict[str, Dict[str, Any]],
        epochs: int = 100,
        batch_size: int = 32,
        initial_seed: int = 42,
        progress_callback: Optional[Callable[[str, int, int, TrainingHistory], None]] = None
    ) -> Dict[str, TrainingHistory]:
        """
        Run training across all requested optimizers.
        """
        all_histories: Dict[str, TrainingHistory] = {}

        for opt_name, cfg in optimizer_configs.items():
            opt_instance = get_optimizer(opt_name, **cfg)
            
            def per_epoch_cb(ep: int, hist: TrainingHistory):
                if progress_callback is not None:
                    progress_callback(opt_name, ep, epochs, hist)

            history = self.train_single_optimizer(
                optimizer=opt_instance,
                epochs=epochs,
                batch_size=batch_size,
                initial_seed=initial_seed,
                epoch_callback=per_epoch_cb
            )
            all_histories[opt_name] = history

        return all_histories

    @staticmethod
    def generate_comparison_dataframe(histories: Dict[str, TrainingHistory]) -> pd.DataFrame:
        """
        Generate summary table formatted as specified in PDF Section B3:
        | Optimizer | Final Train Loss | Final Test Loss | Train Acc. | Test Acc. | Convergence Epoch |
        """
        rows = []
        for opt_name, hist in histories.items():
            final_train_loss = hist.train_losses[-1] if hist.train_losses else np.nan
            final_test_loss = hist.test_losses[-1] if hist.test_losses else np.nan
            final_train_acc = hist.train_accuracies[-1] if hist.train_accuracies else np.nan
            final_test_acc = hist.test_accuracies[-1] if hist.test_accuracies else np.nan
            conv_epoch = hist.compute_convergence_epoch()

            rows.append({
                "Optimizer": opt_name,
                "Final Train Loss": f"{final_train_loss:.4f}" if not np.isnan(final_train_loss) else "NaN",
                "Final Test Loss": f"{final_test_loss:.4f}" if not np.isnan(final_test_loss) else "NaN",
                "Train Acc.": f"{final_train_acc * 100:.2f}%" if not np.isnan(final_train_acc) else "NaN",
                "Test Acc.": f"{final_test_acc * 100:.2f}%" if not np.isnan(final_test_acc) else "NaN",
                "Convergence Epoch": conv_epoch
            })

        df = pd.DataFrame(rows)
        return df
