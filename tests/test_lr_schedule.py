import math

import pytest
import torch


def test_warmup_then_cosine_schedule(train_module):
    factor = train_module._warmup_cosine_factor(warmup_steps=4, total_steps=20, min_lr_ratio=0.1)

    # Linear warmup: step 0..3 -> 0.25, 0.5, 0.75, 1.0
    assert [round(factor(s), 6) for s in range(4)] == [0.25, 0.5, 0.75, 1.0]

    # Peak at the end of warmup, then cosine decay...
    assert factor(4) == pytest.approx(1.0)
    assert factor(12) == pytest.approx(0.55, abs=1e-6)

    # ...down to min_lr_ratio at the final step.
    assert factor(20) == pytest.approx(0.1, abs=1e-6)


def test_warmup_zero_starts_at_peak(train_module):
    factor = train_module._warmup_cosine_factor(warmup_steps=0, total_steps=10, min_lr_ratio=0.0)
    assert factor(0) == pytest.approx(1.0)
    assert factor(10) == pytest.approx(0.0, abs=1e-6)


def test_schedule_is_monotone_nonincreasing_after_warmup(train_module):
    factor = train_module._warmup_cosine_factor(warmup_steps=10, total_steps=60, min_lr_ratio=0.1)
    values = [factor(s) for s in range(10, 61)]
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:]))


def test_schedule_clamps_min_lr_ratio(train_module):
    assert train_module._warmup_cosine_factor(0, 10, 1.5)(10) == pytest.approx(1.0)
    assert train_module._warmup_cosine_factor(0, 10, -1.0)(10) == pytest.approx(0.0)


def test_total_steps_never_below_warmup(train_module):
    # Regression: warmup >= total must not divide by zero or invert.
    factor = train_module._warmup_cosine_factor(warmup_steps=100, total_steps=4, min_lr_ratio=0.1)
    assert math.isfinite(factor(0))
    assert math.isfinite(factor(200))


def test_plateau_detector(train_module):
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(torch.optim.SGD([torch.zeros(1)], lr=0.1))
    cosine = torch.optim.lr_scheduler.LambdaLR(torch.optim.SGD([torch.zeros(1)], lr=0.1), lambda s: 1.0)

    assert train_module._is_plateau_scheduler(plateau) is True
    assert train_module._is_plateau_scheduler(cosine) is False
    assert train_module._is_plateau_scheduler(None) is False
