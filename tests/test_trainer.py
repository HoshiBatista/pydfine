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
from dfine.backends.native import DFINECriterion, SemSegCriterion  # noqa: E402
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


def _seg_batch(batch=2, n=2, num_classes=80):
    # A fixed box + a filled-box instance mask, so the mask branch has a learnable target.
    samples = torch.rand(batch, 3, IMGSZ, IMGSZ)
    targets = []
    for _ in range(batch):
        boxes = torch.rand(n, 4) * 0.3 + 0.35  # cxcywh inside the image
        masks = torch.zeros(n, IMGSZ, IMGSZ, dtype=torch.uint8)
        for i, (cx, cy, w, h) in enumerate(boxes):
            x0, y0 = int((cx - w / 2) * IMGSZ), int((cy - h / 2) * IMGSZ)
            x1, y1 = int((cx + w / 2) * IMGSZ), int((cy + h / 2) * IMGSZ)
            masks[i, y0:y1, x0:x1] = 1  # mask == box interior
        targets.append(
            {"labels": torch.randint(0, num_classes, (n,)), "boxes": boxes, "masks": masks}
        )
    return samples, targets


def _sem_batch(batch=2, num_classes=4):
    # A simple left/right two-class split — easy for the dense head to overfit.
    samples = torch.rand(batch, 3, IMGSZ, IMGSZ)
    sem = torch.zeros(IMGSZ, IMGSZ, dtype=torch.int64)
    sem[:, IMGSZ // 2 :] = 1
    return samples, [{"sem_mask": sem.clone()} for _ in range(batch)]


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


def _encdec_bias_wd(size):
    """Return the weight_decay applied to encoder/decoder bias params for a preset."""
    cfg = DFINEConfig.preset(size, backbone_pretrained=False)
    model = NativeDFINE.from_config(cfg)
    groups = build_param_groups(model, cfg)
    # Map each param id -> its group's weight_decay (None = inherits optimizer default).
    wd_by_id = {id(p): g.get("weight_decay") for g in groups for p in g["params"]}
    for name, p in model.named_parameters():
        # A *non-norm* enc/dec bias is what distinguishes the two schemes (a norm/bn bias
        # is zero-wd under both). e.g. self_attn.in_proj_bias, sampling_offsets.bias.
        is_encdec = "encoder" in name or "decoder" in name
        if (
            p.requires_grad
            and name.endswith("bias")
            and is_encdec
            and "norm" not in name
            and "bn" not in name
        ):
            return wd_by_id[id(p)]
    return None


def test_encdec_bias_zero_wd_matches_upstream_per_size():
    # Upstream N/S/M put encoder/decoder biases in the zero-weight-decay group; L/X don't.
    assert _encdec_bias_wd("n") == 0.0
    assert _encdec_bias_wd("s") == 0.0
    assert _encdec_bias_wd("m") == 0.0
    # L/X: biases inherit the optimizer default (not in the zero-wd group).
    assert _encdec_bias_wd("l") is None
    assert _encdec_bias_wd("x") is None


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


def test_multistep_scheduler_honors_config_milestones():
    from torch.optim.lr_scheduler import MultiStepLR

    model = NativeDFINE.from_config(_cfg())
    cfg = _cfg(scheduler="multistep", lr_milestones=[2], lr_gamma=0.1, epochs=10)
    opt = build_optimizer(model, cfg)
    sched = build_lr_scheduler(opt, cfg)
    assert isinstance(sched, MultiStepLR)
    base = opt.param_groups[-1]["lr"]
    opt.step()
    for _ in range(2):  # step past the milestone at epoch 2
        sched.step()
    assert opt.param_groups[-1]["lr"] == pytest.approx(base * 0.1)


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


# --- seg training: task picks the criterion + the mask/pixel loss optimizes ---


def _mask_loss(stats):
    return sum(v for k, v in stats.items() if "mask" in k)


def test_trainer_selects_criterion_by_task(tmp_path):
    from dfine.train.trainer import Trainer

    seg = Trainer(
        NativeDFINE.from_config(_cfg(task="segment")),
        _cfg(task="segment"),
        device=torch.device("cpu"),
        output_dir=tmp_path / "seg",
        use_ema=False,
        visualize=False,
    )
    assert isinstance(seg.criterion, DFINECriterion) and "masks" in seg.criterion.losses

    ss = Trainer(
        NativeDFINE.from_config(_cfg(task="sem_seg", num_classes=4)),
        _cfg(task="sem_seg", num_classes=4),
        device=torch.device("cpu"),
        output_dir=tmp_path / "ss",
        use_ema=False,
        visualize=False,
    )
    assert isinstance(ss.criterion, SemSegCriterion)


def test_overfit_one_batch_drops_segment_mask_loss():
    torch.manual_seed(0)
    cfg = _cfg(task="segment", lr=1e-3, lr_backbone=1e-3, clip_max_norm=0.1, num_denoising=0)
    model = NativeDFINE.from_config(cfg)
    criterion = DFINECriterion.from_config(cfg)
    optimizer = build_optimizer(model, cfg)
    loader = [_seg_batch(n=2)]  # single fixed batch, reused every epoch
    device = torch.device("cpu")

    first = train_one_epoch(model, criterion, loader, optimizer, device, 0, print_freq=100)
    assert "loss_mask_bce" in first and "loss_mask_dice" in first  # mask terms supervised
    best_total, best_mask = first["loss"], _mask_loss(first)
    for epoch in range(1, 40):
        s = train_one_epoch(model, criterion, loader, optimizer, device, epoch, print_freq=100)
        best_total, best_mask = min(best_total, s["loss"]), min(best_mask, _mask_loss(s))
        assert all(v == v for v in s.values())  # no NaNs
    assert best_total < first["loss"] * 0.5
    assert best_mask < _mask_loss(first)  # the mask branch actually optimizes


def test_overfit_one_batch_drops_sem_seg_loss():
    torch.manual_seed(0)
    cfg = _cfg(task="sem_seg", num_classes=4, lr=1e-3, lr_backbone=1e-3, clip_max_norm=0.1)
    model = NativeDFINE.from_config(cfg)
    criterion = SemSegCriterion.from_config(cfg)
    optimizer = build_optimizer(model, cfg)
    loader = [_sem_batch(num_classes=4)]
    device = torch.device("cpu")

    first = train_one_epoch(model, criterion, loader, optimizer, device, 0, print_freq=100)
    assert {"loss_ce", "loss_dice", "loss_aux"} <= set(first)
    best = first["loss"]
    for epoch in range(1, 40):
        s = train_one_epoch(model, criterion, loader, optimizer, device, epoch, print_freq=100)
        best = min(best, s["loss"])
        assert all(v == v for v in s.values())
    assert best < first["loss"] * 0.5  # the dense pixel loss optimizes


def test_visualizer_tb_logdir(tmp_path):
    from dfine.train.visualizer import TrainingVisualizer

    assert TrainingVisualizer(tmp_path, use_tensorboard=False, plot=False).tb_logdir is None
    v = TrainingVisualizer(tmp_path, use_tensorboard=True, plot=False)
    # tb_logdir points at the tb subdir when tensorboard is installed, else None.
    assert v.tb_logdir in (None, tmp_path / "tb")
    if v.writer is not None:
        assert v.tb_logdir == tmp_path / "tb"
    v.close()


def test_tensorboard_hint_logged_before_train(tmp_path):
    import io
    import logging
    import types

    from dfine.log import LOGGER
    from dfine.train.trainer import Trainer
    from dfine.train.visualizer import TrainingVisualizer

    v = TrainingVisualizer(tmp_path, use_tensorboard=True, plot=False)
    if v.tb_logdir is None:
        pytest.skip("tensorboard not installed — no hint to print")

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    LOGGER.addHandler(handler)
    try:
        # _log_tensorboard_hint only touches self.visualizer — a stub stands in for Trainer.
        Trainer._log_tensorboard_hint(types.SimpleNamespace(visualizer=v))
    finally:
        LOGGER.removeHandler(handler)
        v.close()
    out = buf.getvalue()
    assert "tensorboard --logdir" in out and str(v.tb_logdir) in out and "localhost:6006" in out
