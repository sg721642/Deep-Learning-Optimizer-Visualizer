"""
From SGD to AdamW — Deep Learning Optimizer Visualizer
Dataset loading, preprocessing, standardization, and dynamic metadata.
"""
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class DatasetManager:
    """Manages dataset loading, preprocessing, feature scaling, and train/test splitting."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.raw_data = load_breast_cancer()
        self._prepare_data()

    def _prepare_data(self):
        X = self.raw_data.data.astype(np.float64)
        # Binary target: 1 = benign, 0 = malignant (or original format)
        y = self.raw_data.target.astype(np.float64).reshape(-1, 1)

        # Dynamic sample and feature calculations
        self.total_samples = int(X.shape[0])
        self.num_features = int(X.shape[1])
        self.feature_names = list(self.raw_data.feature_names)
        self.target_names = list(self.raw_data.target_names)

        # Train/Test Split (stratified)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # Feature standardization (fit on train, transform on train and test)
        self.mean = np.mean(X_train_raw, axis=0, keepdims=True)
        self.std = np.std(X_train_raw, axis=0, keepdims=True)
        # Avoid division by zero
        self.std[self.std == 0.0] = 1.0

        self.X_train = (X_train_raw - self.mean) / self.std
        self.X_test = (X_test_raw - self.mean) / self.std
        self.y_train = y_train
        self.y_test = y_test

        self.train_samples = int(self.X_train.shape[0])
        self.test_samples = int(self.X_test.shape[0])

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return standardized (X_train, y_train, X_test, y_test)."""
        return self.X_train, self.y_train, self.X_test, self.y_test

    def get_metadata(self) -> Dict[str, Any]:
        """Return dynamic metadata for UI rendering."""
        return {
            "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
            "total_samples": self.total_samples,
            "num_features": self.num_features,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "test_ratio": self.test_size,
            "positive_class_ratio": float(np.mean(self.y_train)),
            "feature_names": self.feature_names,
            "target_names": self.target_names
        }
