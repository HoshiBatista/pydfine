"""Training progress logging — the console readout, ported from upstream D-FINE.

`MetricLogger`/`SmoothedValue` reproduce D-FINE's `src/misc/logger.py` (itself from
DETR/torchvision) so training prints the same familiar progress line::

    Epoch: [3/72]  [ 120/1850]  eta: 0:18:44  loss: 12.4130 (13.0027)  lr: 0.000250
    time: 0.6091  data: 0.0021  max mem: 5124

The distributed all-gather/synchronize paths are dropped (this library trains
single-process for now); the numbers and formatting are otherwise identical.
"""

from __future__ import annotations

import datetime
import time
from collections import defaultdict, deque
from collections.abc import Iterable

import torch

from ..log import LOGGER, colorstr, fmt_num

__all__ = ["SmoothedValue", "MetricLogger", "ProgressBar"]


class SmoothedValue:
    """Track a series of values, exposing windowed median/avg and global average."""

    def __init__(self, window_size: int = 20, fmt: str | None = None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque: deque[float] = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self) -> float:
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    """Aggregate named `SmoothedValue` meters and stream a progress line per N iters."""

    def __init__(self, delimiter: str = "\t"):
        self.meters: dict[str, SmoothedValue] = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f"{k}={v!r} is not a scalar"
            self.meters[k].update(v)

    def __getattr__(self, attr):
        meters = self.__dict__.get("meters", {})
        if attr in meters:
            return meters[attr]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {attr!r}")

    def __str__(self) -> str:
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def log_every(self, iterable: Iterable, print_freq: int, header: str | None = None):
        """Yield from `iterable`, printing the progress line every `print_freq` steps."""
        i = 0
        header = header or ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        n = len(iterable) if hasattr(iterable, "__len__") else None
        space_fmt = ":" + str(len(str(n))) + "d" if n is not None else ":d"
        cuda = torch.cuda.is_available()
        parts = [
            colorstr("bright_blue", "bold", header) if header else "",
            colorstr("gray", "[{0" + space_fmt + "}/{1}]"),
            colorstr("dim", "eta:") + " {eta}",
            "{meters}",
            colorstr("dim", "time:") + " {time}",
            colorstr("dim", "data:") + " {data}",
        ]
        if cuda:
            parts.append(colorstr("dim", "max mem:") + " {memory:.0f}")
        log_msg = self.delimiter.join(parts)
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or (n is not None and i == n - 1):
                total = n if n is not None else i + 1
                eta_seconds = iter_time.global_avg * (total - i)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                fields = dict(eta=eta, meters=str(self), time=str(iter_time), data=str(data_time))
                if cuda:
                    fields["memory"] = torch.cuda.max_memory_allocated() / MB
                LOGGER.info(log_msg.format(i, n if n is not None else "?", **fields))
            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        per_it = total_time / max(i, 1)
        LOGGER.info(colorstr("dim", f"{header} total time: {total_time_str} ({per_it:.4f} s/it)"))

    def global_avg_dict(self) -> dict[str, float]:
        """`{meter: global_avg}` — the per-epoch summary returned by the train loop."""
        return {k: m.global_avg for k, m in self.meters.items()}


class ProgressBar:
    """An ultralytics-style training progress bar.

    Wraps ``tqdm`` when it is installed for a live single-line bar, and otherwise falls
    back to logging one compact ``desc  [i/N]  eta  <postfix>`` line every ``print_freq``
    steps (plus a final total-time line) so piped/redirected logs stay readable. The
    postfix — set per step via :meth:`set_postfix` with e.g. ``loss``/``lr`` — is the
    *only* per-iteration readout; the full per-term loss breakdown is not printed (it
    still streams to TensorBoard/W&B via the visualizer).
    """

    def __init__(
        self,
        iterable: Iterable,
        *,
        desc: str = "",
        total: int | None = None,
        enabled: bool = True,
        print_freq: int = 50,
    ):
        self.iterable = iterable
        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)
        self.total = total
        self.desc = desc
        self.enabled = enabled
        self.print_freq = max(1, print_freq)
        self._postfix = ""
        self._tqdm = None
        if enabled:
            try:  # optional dependency (in the [train] extra) — degrade gracefully
                from tqdm.auto import tqdm

                self._tqdm = tqdm(
                    total=total,
                    desc=colorstr("bright_blue", "bold", desc),
                    dynamic_ncols=True,
                    leave=True,
                    unit="it",
                )
            except Exception:  # pragma: no cover - tqdm missing/unusable
                self._tqdm = None

    def __iter__(self):
        start = time.time()
        end = start
        it_time = SmoothedValue(fmt="{avg:.4f}")
        i = 0
        for i, obj in enumerate(self.iterable):
            yield obj
            it_time.update(time.time() - end)
            end = time.time()
            if self._tqdm is not None:
                self._tqdm.update(1)
            elif self.enabled and (
                i % self.print_freq == 0 or (self.total is not None and i == self.total - 1)
            ):
                self._log_line(i, it_time)
        if self._tqdm is not None:
            self._tqdm.close()
        elif self.enabled:
            elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
            LOGGER.info(colorstr("dim", f"{self.desc} total time: {elapsed} ({i + 1} it)"))

    def set_postfix(self, **kwargs) -> None:
        """Set the trailing readout shown on the bar (e.g. ``loss=…  lr=…``)."""
        self._postfix = "  ".join(
            f"{colorstr('gray', k)} {colorstr('bright_cyan', 'bold', fmt_num(v))}"
            for k, v in kwargs.items()
        )
        if self._tqdm is not None:
            self._tqdm.set_postfix(
                {k: fmt_num(v) if isinstance(v, float) else v for k, v in kwargs.items()},
                refresh=False,
            )

    def _log_line(self, i: int, it_time: SmoothedValue) -> None:
        total = self.total if self.total is not None else "?"
        eta = ""
        if self.total is not None:
            eta_s = int(it_time.global_avg * (self.total - i))
            eta = "  " + colorstr("dim", "eta:") + " " + str(datetime.timedelta(seconds=eta_s))
        idx = colorstr("gray", f"[{i + 1}/{total}]")
        post = ("  " + self._postfix) if self._postfix else ""
        LOGGER.info(f"{colorstr('bright_blue', 'bold', self.desc)}  {idx}{eta}{post}")
