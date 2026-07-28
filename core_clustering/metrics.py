from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ClassificationResult:
    domain: str
    n_total: int
    n_correct: int
    n_incorrect: int
    accuracy: float
    correct_indices: np.ndarray
    incorrect_indices: np.ndarray

    def compact_dict(self) -> dict:
        """Numbers only -- no indices. This is the exact row shape shared by
        classification_accuracy.csv and run_summary.json's held_out_accuracy."""
        return {
            "domain": self.domain,
            "n_total": self.n_total,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
            "accuracy": self.accuracy,
        }


def evaluate_classification(
    model,
    Y: np.ndarray,
    labels: np.ndarray,
    domain: str,
    device: str = "cpu",
    batch_size: int = 256,
) -> ClassificationResult:
    """Y: (n, n_features, window_size); labels: (n, classes) one-hot. Runs
    the model in eval/no_grad batches and argmaxes predicted vs true class
    -- the same top-1 rule RedLamp used, just against AnomSim's real ground
    truth rather than Loader_aug's own injected pseudo-label bookkeeping.
    """
    model.eval()
    Y_t = torch.from_numpy(Y).float().transpose(2, 1)

    preds = []
    with torch.no_grad():
        for start in range(0, len(Y_t), batch_size):
            batch = Y_t[start : start + batch_size].to(device)
            _, x_out, _ = model(batch)
            preds.append(x_out.argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    true = labels.argmax(axis=1)

    correct_mask = pred == true
    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(~correct_mask)[0]
    n_total = len(true)
    n_correct = len(correct_indices)
    n_incorrect = len(incorrect_indices)
    accuracy = (n_correct / n_total) if n_total else 0.0

    return ClassificationResult(
        domain=domain,
        n_total=n_total,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        accuracy=accuracy,
        correct_indices=correct_indices,
        incorrect_indices=incorrect_indices,
    )
