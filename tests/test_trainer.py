"""Phase-4 training-loop tests: param groups, EMA, warmup, logger, and overfit.

The overfit-one-batch test runs the real native model in training mode through the
criterion and the actual ``train_one_epoch`` loop, and checks the loss drops sharply
on a single fixed batch — the standard "the loop actually optimizes" smoke test. The
matcher needs scipy (train extra), so the whole module is skipped without it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from dfine import DFINEConfig  # noqa: E402
from dfine.backends.native import DFINE as NativeDFINE  # noqa: E402
from dfine.backends.native import DFINECriterion  # noqa: E402
from dfine.train import ModelEMA, SmoothedValue  # noqa: E402
from dfine.train.logger import MetricLogger  # noqa: E402
from dfine.train.scheduler import LinearWarmup, build_lr_scheduler  # noqa: E402
from dfine.train.trainer import (  # noqa: E402
    build_optimizer,
    build_param_groups,
    train_one_epoch,
)

IMGSZ = 320


def _cfg(**kw):
    return DFINEConfig.preset(
        "n", imgsz=IMGSZ, backbone_pretrained=False, freeze_norm=False, freeze_at=-1, **kw
    )


def _batch(batch=2, n=3, num_classes=80):
    samples = torch.rand(batch, 3, IMGSZ, IMGSZ)
    targets = [
        {
            "labels": torch.randint(0, num_classes, (n,)),
            "boxes": torch.rand(n, 4) * 0.4 + 0.3,  # cxcywh, safely inside the image
        }
        for _ in range(batch)
    ]
    return samples, targets


# --- param groups -------------------------------------------------------------


def test_build_param_groups_splits_backbone_and_norms():
    model = NativeDFINE.from_config(_cfg())
    groups = build_param_groups(model, _cfg(lr=1e-3, lr_backbone=1e-4))
    # Backbone group carries the backbone LR; a norm group carries weight_decay=0.
    assert any(g.get("lr") == 1e-4 for g in groups)
    assert any(g.get("weight_decay") == 0.0 for g in groups)
    # Every trainable param is accounted for exactly once.
    grouped = sum(len(g["params"]) for g in groups)
    total = sum(1 for p in model.parameters() if p.requires_grad)
    assert grouped == total


def test_build_optimizer_returns_adamw():
    model = NativeDFINE.from_config(_cfg())
    opt = build_optimizer(model, _cfg())
    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) >= 2


# --- logger / EMA / warmup ----------------------------------------------------


def test_smoothed_value_and_metric_logger():
    logger = MetricLogger(delimiter="  ")
    for v in (10.0, 8.0, 6.0):
        logger.update(loss=v)
    assert logger.meters["loss"].global_avg == pytest.approx(8.0)
    assert "loss" in str(logger)
    sv = SmoothedValue(window_size=3)
    for v in (1.0, 2.0, 3.0):
        sv.update(v)
    assert sv.median == pytest.approx(2.0)
    assert sv.global_avg == pytest.approx(2.0)


def test_model_ema_moves_toward_model():
    model = NativeDFINE.from_config(_cfg())
    ema = ModelEMA(model, decay=0.5, warmups=0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # shift the live model away from the EMA copy
    before = next(iter(ema.module.parameters())).clone()
    ema.update(model)
    after = next(iter(ema.module.parameters()))
    assert not torch.equal(before, after)  # EMA followed the model


def test_linear_warmup_ramps_lr():
    model = NativeDFINE.from_config(_cfg())
    opt = build_optimizer(model, _cfg(lr=1e-3))
    sched = build_lr_scheduler(opt, _cfg())
    warmup = LinearWarmup(sched, warmup_duration=10)
    first = opt.param_groups[-1]["lr"]
    for _ in range(5):
        warmup.step()
    mid = opt.param_groups[-1]["lr"]
    assert mid > first
    assert not warmup.finished()
    for _ in range(10):
        warmup.step()
    assert warmup.finished()


def test_flatcosine_scheduler_is_flat_then_decays():
    model = NativeDFINE.from_config(_cfg())
    opt = build_optimizer(model, _cfg())
    cfg = _cfg(epochs=10, no_aug_epoch=4)
    sched = build_lr_scheduler(opt, cfg)
    base = opt.param_groups[-1]["lr"]
    opt.step()  # avoid the "scheduler stepped before optimizer" warning
    for _ in range(6):  # flat region (epochs - no_aug_epoch)
        sched.step()
    flat = opt.param_groups[-1]["lr"]
    assert flat == pytest.approx(base)
    for _ in range(4):
        sched.step()
    assert opt.param_groups[-1]["lr"] < base  # cosine tail decayed


# --- the loop actually optimizes ----------------------------------------------


def test_overfit_one_batch_drops_loss():
    torch.manual_seed(0)
    # Denoising off keeps the objective clean (no noised-GT terms) so a single fixed
    # batch overfits decisively — this checks the loop optimizes, not convergence speed.
    # lr=1e-3 stays in a stable regime (lr=2e-3 overshoots and the loss oscillates, so
    # the *final* epoch lands unpredictably across platforms/torch builds).
    cfg = _cfg(lr=1e-3, lr_backbone=1e-3, clip_max_norm=0.1, num_denoising=0)
    model = NativeDFINE.from_config(cfg)
    criterion = DFINECriterion.from_config(cfg)
    optimizer = build_optimizer(model, cfg)
    loader = [_batch(n=2)]  # a single fixed batch, reused every epoch
    device = torch.device("cpu")

    first = train_one_epoch(model, criterion, loader, optimizer, device, 0, print_freq=100)
    best = first["loss"]
    for epoch in range(1, 60):
        stats = train_one_epoch(model, criterion, loader, optimizer, device, epoch, print_freq=100)
        best = min(best, stats["loss"])
        assert all(v == v for v in stats.values())  # no NaNs, every epoch
    # Overfitting a single batch should cut the total loss substantially. Check the best
    # loss reached, not the last epoch's, so a bit of tail wobble can't flake the test.
    assert best < first["loss"] * 0.5
