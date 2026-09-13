import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oilfield.events.cnn import WellEventCNN, make_batch, predict_cnn, sensor_scale, train_cnn  # noqa: E402


def toy_sequences(n_windows=256, length=60, seed=0):
    """Class 1 windows ramp up, class 0 windows are flat noise; one NaN-heavy sensor."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n_windows)
    values = np.empty((n_windows * length, 2), dtype=np.float32)
    for i, label in enumerate(y):
        t = np.arange(length)
        values[i * length:(i + 1) * length, 0] = 150 + rng.normal(0, 0.05, length) + (0.05 * t if label else 0)
        values[i * length:(i + 1) * length, 1] = np.nan
    starts = np.arange(n_windows) * length
    return values, starts, y.astype(np.int64)


def test_batch_shape_masks_and_centering():
    values, starts, _ = toy_sequences(4)
    x = make_batch(values, starts, 60, np.array([1.0, 1.0], dtype=np.float32))
    assert x.shape == (4, 4, 60)
    assert np.allclose(x[:, 0].mean(axis=1), 0, atol=1e-4)   # window mean removed
    assert (x[:, 2] == 1).all() and (x[:, 3] == 0).all()     # validity channels
    assert (x[:, 1] == 0).all()                               # missing sensor zero-filled


def test_forward_and_short_training_separates_toy_classes():
    values, starts, y = toy_sequences()
    model = WellEventCNN(n_sensors=2, n_classes=3, width=8)
    assert model(torch.zeros(5, 4, 60)).shape == (5, 3)

    scale = sensor_scale(np.array([[0.5, np.nan], [1.0, np.nan]]))
    assert scale[1] == 1.0
    params = {"epochs": 4, "batch_size": 32, "lr": 3e-3, "width": 8}
    model = train_cnn(values, starts, y, length=60, scale=scale, n_classes=3, params=params,
                      device=torch.device("cpu"))
    proba = predict_cnn(model, values, starts, length=60, scale=scale)
    assert proba.shape == (len(y), 3)
    assert (proba.argmax(1) == y).mean() > 0.9
