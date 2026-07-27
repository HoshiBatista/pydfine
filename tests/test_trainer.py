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
from dfine.train import ModelEMA, ProgressBar, SmoothedValue  # noqa: E402
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


def test_progress_bar_yields_all_items_and_takes_postfix():
    bar = ProgressBar(list(range(5)), desc="Epoch: [0/1]", print_freq=2)
    seen = []
    for i, x in enumerate(bar):
        seen.append(x)
        bar.set_postfix(loss=1.0 / (i + 1), lr=1e-4)  # must not raise on either backend
    assert seen == [0, 1, 2, 3, 4]
    assert bar.total == 5


def test_progress_bar_fallback_logs_without_tqdm(monkeypatch):
    """With tqdm unavailable, the bar degrades to compact periodic log lines."""
    import builtins

    real_import = builtins.__import__

    def no_tqdm(name, *args, **kwargs):
        if name.startswith("tqdm"):
            raise ImportError("tqdm disabled for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tqdm)
    bar = ProgressBar(range(4), desc="Epoch: [0/1]", print_freq=1)
    assert bar._tqdm is None  # fell back
    assert list(bar) == [0, 1, 2, 3]


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


def test_fit_writes_last_best_and_periodic_checkpoints(tmp_path):
    """`fit` always writes last.pth; best.pth on metric improvement; periodic snapshots
    under weights/ every `checkpoint_freq` epochs (and never on freq <= 0)."""
    from dfine.train.trainer import Trainer

    # A val_fn returning an improving AP each epoch → best.pth rewritten every epoch.
    scores = iter([0.1, 0.2, 0.3, 0.4])

    def val_fn(module, loader):
        return {"AP": next(scores)}

    cfg = _cfg(checkpoint_freq=2, num_denoising=0)
    trainer = Trainer(
        NativeDFINE.from_config(cfg),
        cfg,
        device=torch.device("cpu"),
        output_dir=tmp_path / "run",
        use_ema=False,
        use_amp=False,
        visualize=False,
    )
    loader = [_batch(n=2)]
    trainer.fit(loader, epochs=4, val_loader=loader, val_fn=val_fn)

    run = tmp_path / "run"
    assert (run / "last.pth").exists()
    assert (run / "best.pth").exists()
    # checkpoint_freq=2 over 4 epochs (0..3) → epochs 1 and 3 snapshot (3 is also the last).
    saved = sorted(p.name for p in (run / "weights").glob("*.pth"))
    assert saved == ["epoch1.pth", "epoch3.pth"]
    # A saved checkpoint carries the full resumable state (model + optimizer + epoch).
    ckpt = torch.load(run / "weights" / "epoch1.pth", map_location="cpu", weights_only=False)
    assert {"epoch", "model", "optimizer", "lr_scheduler"} <= set(ckpt)


def test_fit_resume_continues_from_saved_epoch(tmp_path):
    """`fit(resume=…)` restores model/optimizer/scheduler/epoch and picks up where it left
    off — a 2-epoch run resumed to 4 finishes at epoch 3 with the scheduler fully advanced."""
    from dfine.train.trainer import Trainer

    torch.manual_seed(0)
    cfg = _cfg(num_denoising=0, warmup_iters=0, scheduler="multistep", lr_milestones=[1])
    trainer = Trainer(
        NativeDFINE.from_config(cfg),
        cfg,
        device=torch.device("cpu"),
        output_dir=tmp_path / "run",
        use_ema=False,
        use_amp=False,
        visualize=False,
    )

    loader = [_batch(n=2)]
    trainer.fit(loader, epochs=2)  # epochs 0,1 → last.pth at epoch 1
    assert trainer.resume_from(tmp_path / "run" / "last.pth") == 2  # picks up at saved+1

    trainer.fit(loader, epochs=4, resume=True)  # resume=True → run/last.pth, runs epochs 2,3
    last = torch.load(tmp_path / "run" / "last.pth", map_location="cpu", weights_only=False)
    assert last["epoch"] == 3  # ran through the final epoch

    # Scheduler advanced all 4 steps; the milestone at epoch 1 fired once → lr *= gamma.
    lr = trainer.optimizer.param_groups[-1]["lr"]
    assert lr == pytest.approx(cfg.lr * cfg.lr_gamma)


def test_resume_from_missing_file_raises(tmp_path):
    from dfine.train.trainer import Trainer

    t = Trainer(
        NativeDFINE.from_config(_cfg()),
        _cfg(),
        device=torch.device("cpu"),
        output_dir=tmp_path / "run",
        use_ema=False,
        visualize=False,
    )
    with pytest.raises(FileNotFoundError):
        t.resume_from(tmp_path / "nope.pth")


def test_fit_no_periodic_checkpoints_when_freq_disabled(tmp_path):
    from dfine.train.trainer import Trainer

    cfg = _cfg(checkpoint_freq=-1, num_denoising=0)  # the default: periodic snapshots off
    trainer = Trainer(
        NativeDFINE.from_config(cfg),
        cfg,
        device=torch.device("cpu"),
        output_dir=tmp_path / "run",
        use_ema=False,
        use_amp=False,
        visualize=False,
    )
    trainer.fit([_batch(n=2)], epochs=2)
    assert (tmp_path / "run" / "last.pth").exists()
    assert not (tmp_path / "run" / "weights").exists()  # nothing periodic written


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
    # The single-batch overfit is noisy/non-monotonic (loss can spike between epochs), and CPU
    # numerics differ across torch/Python builds, so the min-over-epochs ratio varies by ~0.2
    # platform to platform. 60 epochs + a 0.6 bound keep this a robust "loss falls substantially"
    # check (a healthy run reaches ~0.15-0.35; a broken/disconnected head stays near 1.0).
    for epoch in range(1, 60):
        s = train_one_epoch(model, criterion, loader, optimizer, device, epoch, print_freq=100)
        best_total, best_mask = min(best_total, s["loss"]), min(best_mask, _mask_loss(s))
        assert all(v == v for v in s.values())  # no NaNs
    assert best_total < first["loss"] * 0.6
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


@pytest.mark.parametrize(
    "metric_key, value",
    [("AP", 0.42), ("mAP_50_95_mask", 0.31), ("mIoU", 0.77)],
)
def test_loss_curve_panel_tracks_task_primary_metric(tmp_path, metric_key, value):
    # The progress curve's second panel must follow the task's primary metric — detection AP,
    # instance-seg mask AP, or sem_seg mIoU — not just "AP", so seg runs get a metric panel too.
    pytest.importorskip("matplotlib")
    from dfine.train.visualizer import TrainingVisualizer

    v = TrainingVisualizer(tmp_path, use_tensorboard=False, plot=True)
    v.log_step(0, total_loss=1.0, lrs=[1e-4], loss_dict={"loss": 1.0})
    v.log_epoch(0, {"loss": 1.0}, {metric_key: value})

    assert v._metric_key == metric_key  # locked onto this task's metric
    assert v._epoch_metric == [value]
    assert (tmp_path / "loss_curve.png").exists()  # the metric panel rendered
    v.close()


def test_loss_curve_panel_ignores_metricless_and_foreign_keys(tmp_path):
    # A val dict without any known primary metric (or from a different task once locked) must
    # not pollute the tracked series.
    pytest.importorskip("matplotlib")
    from dfine.train.visualizer import TrainingVisualizer

    v = TrainingVisualizer(tmp_path, use_tensorboard=False, plot=True)
    v.log_step(0, total_loss=1.0, lrs=[1e-4], loss_dict={"loss": 1.0})
    v.log_epoch(0, {"loss": 1.0}, {"pixel_acc": 0.9})  # no primary metric present
    assert v._metric_key is None and v._epoch_metric == []

    v.log_epoch(1, {"loss": 0.9}, {"mIoU": 0.5})  # locks onto mIoU
    v.log_epoch(2, {"loss": 0.8}, {"AP": 0.6})  # foreign key ignored after lock
    assert v._metric_key == "mIoU" and v._epoch_metric == [0.5]
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
