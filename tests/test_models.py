import torch

from core_clustering.models import ConvAEC, ModelConfig, NonLinClassifier, predict_proba


def make_tiny_config(**overrides):
    defaults = dict(
        n_features=1,
        n_time=100,
        classes=3,
        num_filters=[8, 8],
        embedding_dim=4,
        kernel_size=4,
        dropout=0.0,
        normalization="batch",
        stride=2,
        padding=2,
        classifier_dim=4,
        c_loss_ratio=0.1,
        apply_anomaly_mask=True,
        label_smoothing=True,
        alpha=0.1,
        beta=0.01,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_forward_pass_output_shapes():
    config = make_tiny_config()
    model = ConvAEC(config)
    x = torch.randn(8, 100, 1)
    x_hat, x_out, x_enc = model(x)
    assert x_hat.shape == x.shape
    assert x_out.shape == (8, 3)
    assert x_enc.shape == (8, config.embedding_dim, 1)


def test_calculate_loss_runs_without_nan():
    config = make_tiny_config()
    model = ConvAEC(config)
    x = torch.randn(8, 100, 1)
    x_hat, x_out, x_enc = model(x)
    label = torch.zeros(8, 3)
    label[torch.arange(8), torch.randint(0, 3, (8,))] = 1.0
    anomaly_mask = torch.ones(8, 100, 1)
    loss, loss_ae, loss_c = model.calculate_loss(x, x_hat, label, x_out, anomaly_mask, epoch=0)
    assert not torch.isnan(loss).any()
    assert not torch.isnan(loss_ae).any()
    assert not torch.isnan(loss_c).any()


def test_calculate_loss_does_not_mutate_caller_label():
    config = make_tiny_config()
    model = ConvAEC(config)
    x = torch.randn(8, 100, 1)
    x_hat, x_out, x_enc = model(x)
    label = torch.zeros(8, 3)
    label[torch.arange(8), torch.randint(0, 3, (8,))] = 1.0
    label_before = label.clone()
    anomaly_mask = torch.ones(8, 100, 1)
    model.calculate_loss(x, x_hat, label, x_out, anomaly_mask, epoch=0)
    torch.testing.assert_close(label, label_before)


def test_classifier_output_is_raw_logits_not_softmax():
    torch.manual_seed(0)
    classifier = NonLinClassifier(d_in=8, n_class=3, d_hidd=4, norm="batch")
    x = torch.randn(6, 8)
    out = classifier(x)
    assert not torch.allclose(out.sum(dim=1), torch.ones(out.shape[0]), atol=1e-4)
    assert (out < 0).any()


def test_predict_proba_rows_sum_to_one():
    logits = torch.randn(6, 3)
    proba = predict_proba(logits)
    torch.testing.assert_close(proba.sum(dim=1), torch.ones(6), atol=1e-5, rtol=1e-5)
    assert (proba >= 0).all()
