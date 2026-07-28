import numpy as np
import torch
import torch.nn as nn

from core_clustering.metrics import evaluate_classification


class _StubModel(nn.Module):
    """Returns fixed logits regardless of input, so correctness is fully
    determined by the true labels passed to evaluate_classification."""

    def __init__(self, fixed_logits: torch.Tensor):
        super().__init__()
        self.fixed_logits = fixed_logits

    def forward(self, x):
        batch = x.shape[0]
        x_out = self.fixed_logits[:batch]
        return x, x_out, x


def test_evaluate_classification_counts_correct_and_incorrect():
    # 5 windows, 3 classes. Model always predicts class 0.
    fixed_logits = torch.tensor([[5.0, 0.0, 0.0]] * 5)
    model = _StubModel(fixed_logits)

    Y = np.zeros((5, 1, 10), dtype=np.float32)
    # True classes: 0,0,0,1,2 -> model (always predicts 0) gets first 3 right.
    labels = np.zeros((5, 3), dtype=np.float32)
    true_classes = [0, 0, 0, 1, 2]
    for i, c in enumerate(true_classes):
        labels[i, c] = 1.0

    result = evaluate_classification(model, Y, labels, domain="test_domain", device="cpu", batch_size=2)

    assert result.domain == "test_domain"
    assert result.n_total == 5
    assert result.n_correct == 3
    assert result.n_incorrect == 2
    assert result.accuracy == 0.6
    assert sorted(result.correct_indices.tolist()) == [0, 1, 2]
    assert sorted(result.incorrect_indices.tolist()) == [3, 4]


def test_compact_dict_has_no_indices():
    fixed_logits = torch.tensor([[5.0, 0.0]] * 4)
    model = _StubModel(fixed_logits)
    Y = np.zeros((4, 1, 10), dtype=np.float32)
    labels = np.zeros((4, 2), dtype=np.float32)
    labels[:, 0] = 1.0

    result = evaluate_classification(model, Y, labels, domain="d", device="cpu")
    compact = result.compact_dict()

    assert set(compact.keys()) == {"domain", "n_total", "n_correct", "n_incorrect", "accuracy"}
