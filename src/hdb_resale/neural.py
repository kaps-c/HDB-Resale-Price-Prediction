"""Optional, bounded PyTorch experiments with validation-based checkpoints."""

import logging

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn


class FeatureSequenceLSTM(nn.Module):
    """Retain the original feature-as-timestep idea, not a temporal forecast."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=4, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(64, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(inputs.unsqueeze(-1))
        return self.output(self.dropout(sequence[:, -1, :])).squeeze(-1)


def _network(architecture: str, width: int) -> nn.Module:
    if architecture == "lstm":
        return FeatureSequenceLSTM()
    return nn.Sequential(
        nn.Linear(width, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Flatten(0),
    )


def _tensor(values: np.ndarray) -> torch.Tensor:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return torch.as_tensor(np.asarray(values, dtype=np.float32))


class NeuralRegressor(RegressorMixin, BaseEstimator):
    """Small sklearn-compatible predictor; checkpoint tensors stay on the CPU."""

    def __init__(
        self,
        architecture: str = "mlp",
        epochs: int = 20,
        patience: int = 5,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        seed: int = 42,
        device: str = "auto",
        threads: int = 4,
    ) -> None:
        self.architecture = architecture
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = device
        self.threads = threads

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        *,
        validation_data: tuple[np.ndarray, np.ndarray],
    ) -> "NeuralRegressor":
        """Fit with explicit chronological validation, never a random holdout."""
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        device = (
            "cuda"
            if self.device == "auto" and torch.cuda.is_available()
            else self.device
        )
        device = "cpu" if device == "auto" else device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; choose --device cpu")
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        self.n_features_in_ = inputs.shape[1]
        self.target_mean_ = float(np.mean(targets))
        self.target_scale_ = max(float(np.std(targets)), 1.0)
        x = _tensor(inputs)
        y = _tensor((np.asarray(targets) - self.target_mean_) / self.target_scale_)
        x_val, y_val = validation_data
        x_val = _tensor(x_val)
        y_val = np.asarray(y_val)
        network = _network(self.architecture, self.n_features_in_).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        generator = torch.Generator().manual_seed(self.seed)
        best_mae, stale = float("inf"), 0
        self.history_ = []
        self.weights_ = None
        for epoch in range(self.epochs):
            network.train()
            order = torch.randperm(len(x), generator=generator)
            total_loss = 0.0
            for indices in order.split(self.batch_size):
                optimizer.zero_grad()
                loss = criterion(network(x[indices].to(device)), y[indices].to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite neural training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
            network.eval()
            with torch.no_grad():
                normalized = torch.cat(
                    [
                        network(batch.to(device)).cpu()
                        for batch in x_val.split(self.batch_size)
                    ]
                ).numpy()
            predicted = normalized * self.target_scale_ + self.target_mean_
            mae = float(np.mean(np.abs(predicted - y_val)))
            if not np.isfinite(mae):
                raise ValueError("Non-finite neural validation error")
            self.history_.append(
                {
                    "epoch": epoch + 1,
                    "train_mse_scaled": total_loss / len(x),
                    "validation_mae_sgd": mae,
                }
            )
            logging.getLogger(__name__).info(
                "%s epoch %d/%d: validation MAE SGD %.2f",
                self.architecture,
                epoch + 1,
                self.epochs,
                mae,
            )
            if mae < best_mae:
                best_mae, stale = mae, 0
                self.best_epoch_ = epoch + 1
                self.weights_ = {
                    key: value.detach().cpu().clone()
                    for key, value in network.state_dict().items()
                }
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.device_used_ = device
        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Reconstruct on CPU for portable, batched prediction."""
        if self.weights_ is None:
            raise ValueError("Neural checkpoint weights have not been loaded")
        torch.set_num_threads(self.threads)
        network = _network(self.architecture, self.n_features_in_)
        network.load_state_dict(self.weights_)
        network.eval()
        with torch.no_grad():
            normalized = torch.cat(
                [network(batch) for batch in _tensor(inputs).split(self.batch_size)]
            ).numpy()
        return normalized * self.target_scale_ + self.target_mean_
