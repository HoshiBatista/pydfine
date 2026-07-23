"""TS3: SemSegCriterion — CE + soft Dice (+ aux CE), ignore_index handling.

Runs on synthetic logits/label maps (no dataset). The from_config test needs the
preset machinery; the loss math needs only torch.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dfine.backends.native import SemSegCriterion  # noqa: E402
from dfine.config import DFINEConfig  # noqa: E402

C, H, W = 19, 16, 16


def _outputs(batch=2, aux=True, requires_grad=False):
    logits = torch.randn(batch, C, H, W, requires_grad=requires_grad)
    out = {"sem_seg_logits": logits}
    if aux:
        out["sem_seg_logits_aux"] = torch.randn(batch, C, H, W, requires_grad=requires_grad)
    return out


def _targets(batch=2, ignore=255, fill=None):
    return [
        {"sem_mask": torch.randint(0, C, (H, W)) if fill is None else torch.full((H, W), fill)}
        for _ in range(batch)
    ]


def test_from_config_weights_and_ignore_index():
    c = SemSegCriterion.from_config(DFINEConfig.preset("n", task="sem_seg", num_classes=C))
    assert c.weight_dict == {"loss_ce": 1.0, "loss_dice": 1.0, "loss_aux": 0.4}
    assert c.num_classes == C and c.ignore_index == 255


def test_forward_ce_dice_aux_finite_and_differentiable():
    torch.manual_seed(0)
    crit = SemSegCriterion.from_config(DFINEConfig.preset("n", task="sem_seg", num_classes=C))
    out = _outputs(aux=True, requires_grad=True)
    losses = crit(out, _targets())
    assert set(losses) == {"loss_ce", "loss_dice", "loss_aux"}
    assert all(torch.isfinite(v) for v in losses.values())

    sum(losses.values()).backward()
    assert out["sem_seg_logits"].grad is not None
    assert float(out["sem_seg_logits"].grad.abs().sum()) > 0


def test_no_aux_logits_skips_aux_loss():
    crit = SemSegCriterion.from_config(DFINEConfig.preset("n", task="sem_seg", num_classes=C))
    losses = crit(_outputs(aux=False), _targets())
    assert set(losses) == {"loss_ce", "loss_dice"}  # no aux term when the decoder omits it


def test_all_ignore_batch_is_zero_loss():
    crit = SemSegCriterion.from_config(DFINEConfig.preset("n", task="sem_seg", num_classes=C))
    out = _outputs(aux=True, requires_grad=True)
    losses = crit(out, _targets(fill=255))  # every pixel ignored
    assert all(float(v) == 0.0 for v in losses.values())
    # still differentiable (zero grad), so a fully-padded batch never breaks the step.
    sum(losses.values()).backward()


def test_ignore_index_excludes_pixels_from_loss():
    # Logits are confident+correct on the valid pixels but confidently WRONG on the top
    # rows; marking those rows ignore_index must drive CE to ~0 (they don't contribute).
    crit = SemSegCriterion(
        weight_dict={"loss_ce": 1.0, "loss_dice": 1.0, "loss_aux": 0.4}, num_classes=C
    )
    gt = torch.randint(0, C, (H, W))
    logits = torch.full((1, C, H, W), -10.0)
    logits[0, gt, torch.arange(H)[:, None], torch.arange(W)[None, :]] = 10.0  # confident + correct
    logits[0, :, :4] = 0.0  # top rows: uninformative (wrong) predictions

    tgt = gt.clone()
    with_wrong_rows = crit({"sem_seg_logits": logits}, [{"sem_mask": tgt}])["loss_ce"]
    tgt[:4] = 255  # ignore exactly the rows the model gets wrong
    ignored = crit({"sem_seg_logits": logits}, [{"sem_mask": tgt}])["loss_ce"]

    assert float(ignored) < 1e-3 < float(with_wrong_rows)  # ignored rows removed from CE
