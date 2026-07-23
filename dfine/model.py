"""The public, backend-agnostic ``DFINE`` class.

This is the headline API: build a fully-typed model from Python params (a preset
plus any overrides — no YAML), optionally load released weights, and ``predict``.
It wraps the native backend (assembled model + postprocessor) but never leaks
backend details through its kwargs.

    from dfine import DFINE

    model = DFINE(size="l", num_classes=80)     # architecture (ImageNet backbone)
    model.load("dfine-l")                        # released COCO weights
    results = model.predict("street.jpg", conf=0.4)
    results[0].save("out.jpg")

    model = DFINE.from_pretrained("dfine-s")     # one-liner: build + download + load
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from .config import DFINEConfig
from .results import Boxes, Masks, Results, SemSeg

__all__ = ["DFINE"]


def _cleanup_masks(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Zero out mask pixels outside each detection's box (masks ``[N,H,W]``, boxes xyxy)."""
    if masks.numel() == 0:
        return masks
    n, h, w = masks.shape
    ys = torch.arange(h, device=masks.device)[None, :, None]
    xs = torch.arange(w, device=masks.device)[None, None, :]
    x1, y1, x2, y2 = boxes.T[:, :, None, None]
    inside = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
    return masks & inside


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _coco_names() -> dict[int, str]:
    from .backends.native.coco import mscoco_category2name

    return dict(enumerate(mscoco_category2name.values()))


def _build_names(cfg: DFINEConfig) -> dict[int, str]:
    if cfg.class_names:
        return dict(enumerate(cfg.class_names))
    if cfg.num_classes == 80:
        return _coco_names()
    return {i: f"class_{i}" for i in range(cfg.num_classes)}


def _to_pil(item) -> Image.Image:
    """Coerce a path / PIL image / RGB HWC array into an RGB ``PIL.Image``."""
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, (str, os.PathLike)):
        return Image.open(item).convert("RGB")
    try:
        import numpy as np

        if isinstance(item, np.ndarray):
            return Image.fromarray(item).convert("RGB")
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"Unsupported image source: {type(item).__name__}")


def _load_images(source) -> list[Image.Image]:
    if isinstance(source, (list, tuple)):
        return [_to_pil(s) for s in source]
    return [_to_pil(source)]


def _require_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "Video I/O needs OpenCV — install with `pip install pydfine[video]`."
        ) from exc


def _train_worker(rank: int, world_size: int, config, init_weights, kwargs) -> None:
    """DDP worker entry point (module-level so ``mp.spawn`` can pickle it).

    Rebuilds the model from ``config`` on this rank's device, loads the launcher's
    snapshot, joins the process group, and runs the training loop; rank 0 writes the
    checkpoints/logs the launcher later reloads.
    """
    from .train.distributed import cleanup_distributed, setup_distributed

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    setup_distributed()

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    try:
        model = DFINE(config=config, device=device)
        if init_weights is not None:
            model.load(init_weights)
        model._fit(train_loader=None, val_loader=None, val_fn=None, **kwargs)
    finally:
        cleanup_distributed()


class DFINE:
    """Config-first D-FINE detector with an ultralytics-style ``predict``."""

    def __init__(
        self,
        size: str | None = None,
        *,
        config: DFINEConfig | None = None,
        weights: str | os.PathLike | None = None,
        device: str | torch.device | None = None,
        **params,
    ):
        if config is not None:
            if size is not None or params:
                raise ValueError("Pass either `config=` or `size=`/kwargs, not both.")
            self.config = config
        else:
            self.config = DFINEConfig.preset(size, **params) if size else DFINEConfig(**params)
        self.device = _resolve_device(device)
        self.names = _build_names(self.config)

        from .backends.native import DFINE as _NativeDFINE

        self.model = _NativeDFINE.from_config(self.config).to(self.device).eval()
        if self.config.task == "sem_seg":
            from .backends.native import SemSegPostProcessor

            self.postprocessor = SemSegPostProcessor.from_config(self.config).to(self.device).eval()
        else:
            from .backends.native import DFINEPostProcessor

            self.postprocessor = DFINEPostProcessor.from_config(self.config).to(self.device).eval()

        if weights is not None:
            self.load(weights)

    @classmethod
    def from_pretrained(
        cls, name: str, device: str | torch.device | None = None, **overrides
    ) -> DFINE:
        """Build a model matching a released checkpoint and load its weights.

        ``name`` is a catalogue entry (``"dfine-s"``, ``"dfine-l-obj365"`` ...); the
        size and ``num_classes`` are taken from it. See ``dfine models``.
        """
        from .registry import resolve

        spec = resolve(name)
        model = cls(
            size=spec.size,
            device=device,
            num_classes=spec.num_classes,
            backbone_pretrained=False,
            **overrides,
        )
        model.load(name)
        return model

    def load(self, weights: str | os.PathLike, use_ema: bool = True) -> DFINE:
        """Load weights into the model, in place.

        ``weights`` is either a catalogue name (downloaded + cached) or a local
        ``.pth`` path. Returns ``self`` for chaining: ``DFINE(size="s").load("dfine-s")``.
        """
        from .backends.native.loader import load_checkpoint
        from .downloads import download_weights
        from .registry import CHECKPOINTS

        if isinstance(weights, str) and weights.lower() in CHECKPOINTS:
            path = download_weights(weights.lower())
        else:
            path = Path(weights)
            if not path.exists():
                raise FileNotFoundError(
                    f"{weights!r} is neither a known checkpoint name nor an existing file."
                )
        load_checkpoint(self.model, path, use_ema=use_ema, strict=True)
        self.model.to(self.device)
        return self

    @torch.no_grad()
    def predict(
        self,
        source,
        conf: float = 0.25,
        imgsz: int | None = None,
        mask_thresh: float = 0.5,
    ) -> list[Results]:
        """Detect objects in ``source`` (path / PIL / array, or a list of them).

        Returns one :class:`~dfine.results.Results` per image; boxes are in the
        original pixel scale. ``conf`` drops low-scoring detections. For a
        ``task="segment"`` model, each result also carries per-instance
        :class:`~dfine.results.Masks` (original scale), thresholded at ``mask_thresh``.
        For a ``task="sem_seg"`` model each result instead carries a
        :class:`~dfine.results.SemSeg` label map (uint8, original scale) and no boxes.
        """
        images = _load_images(source)
        size = imgsz or self.config.imgsz
        if size != self.config.imgsz:
            raise ValueError(
                f"predict(imgsz={size}) must equal the model's imgsz ({self.config.imgsz}): the "
                "encoder's positional embeddings are precomputed for that resolution. Build the "
                f"model at this size instead — DFINE(size=..., imgsz={size})."
            )
        transform = T.Compose([T.Resize((size, size)), T.ToTensor()])

        batch = torch.stack([transform(im) for im in images]).to(self.device)
        orig_sizes = torch.tensor([[im.width, im.height] for im in images], device=self.device)

        outputs = self.model(batch)
        if self.config.task == "sem_seg":
            label_maps = self.postprocessor(outputs, orig_sizes)
            return [self._to_semseg_results(im, m) for im, m in zip(images, label_maps)]

        detections = self.postprocessor(outputs, orig_sizes)
        pred_masks = outputs.get("pred_masks")
        return [
            self._to_results(
                im, det, conf, None if pred_masks is None else pred_masks[b], mask_thresh
            )
            for b, (im, det) in enumerate(zip(images, detections))
        ]

    __call__ = predict

    def _to_results(
        self,
        image: Image.Image,
        det: dict,
        conf: float,
        pred_masks: torch.Tensor | None = None,
        mask_thresh: float = 0.5,
    ) -> Results:
        scores, labels, boxes = det["scores"], det["labels"], det["boxes"]
        keep = scores > conf
        kept_boxes = boxes[keep]
        boxes_obj = Boxes(
            xyxy=kept_boxes.cpu(),
            conf=scores[keep].cpu(),
            cls=labels[keep].cpu(),
        )

        masks_obj = None
        if pred_masks is not None:
            qidx = det["query_index"][keep]
            m = pred_masks[qidx]
            if m.numel():
                h0, w0 = image.height, image.width
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(0).float(), size=(h0, w0), mode="bilinear", align_corners=False
                ).squeeze(0)
                binm = m >= mask_thresh
                binm = _cleanup_masks(binm, kept_boxes.round().long())
            else:
                binm = torch.zeros((0, image.height, image.width), dtype=torch.bool)
            masks_obj = Masks(binm.cpu())

        return Results(image, boxes_obj, self.names, masks=masks_obj)

    def _to_semseg_results(self, image: Image.Image, label_map: torch.Tensor) -> Results:
        """Wrap a ``[H0, W0]`` uint8 label map (original scale) in a boxless Results."""
        empty = Boxes(
            xyxy=torch.zeros((0, 4)),
            conf=torch.zeros((0,)),
            cls=torch.zeros((0,), dtype=torch.long),
        )
        return Results(image, empty, self.names, sem_seg=SemSeg(label_map.cpu()))

    def train(
        self,
        train_loader=None,
        epochs: int | None = None,
        *,
        data: str | os.PathLike | None = None,
        batch_size: int = 4,
        num_workers: int = 4,
        augment: bool = True,
        remap_mscoco_category: bool = False,
        devices: int | None = None,
        val_loader=None,
        val_fn=None,
        output_dir: str = "runs/train",
        use_wandb: bool = False,
        visualize: bool = True,
    ):
        """Fine-tune the model (Phase 4).

        Provide the data one of two ways:

        * ``data="path/to/coco"`` — a standard COCO dataset root (``train2017/`` +
          ``annotations/instances_train2017.json``, optional ``val2017/``). The train
          loader (full two-phase augmentation + multi-scale) and, if present, a val
          loader are built for you via
          :func:`~dfine.train.dataset.build_coco_dataloaders`. ``batch_size``,
          ``num_workers``, ``augment`` and ``remap_mscoco_category`` tune that build
          (set ``remap_mscoco_category=True`` for stock 80-class MS-COCO ids). For a
          ``task="segment"`` / ``"sem_seg"`` model, ``data=`` is instead a YOLO-style root
          (``images/`` + ``labels/``: polygon ``.txt`` for segment, class-id ``.png`` for
          sem_seg) built via :func:`~dfine.train.seg_dataset.build_seg_dataloader`; seg has
          no auto val eval yet — pass ``val_loader``/``val_fn`` to evaluate.
        * ``train_loader=...`` — a ready dataloader yielding ``(samples, targets)``
          batches: ``samples`` a float ``BCHW`` image tensor, each ``target`` a dict
          with ``labels`` (``LongTensor``) and ``boxes`` (``cxcywh``, normalized).

        **Multi-GPU:** pass ``devices=N`` to train on ``N`` GPUs — this call becomes the
        launcher and spawns one DDP worker per GPU (no ``torchrun`` needed); it requires
        ``data=`` (in-memory loaders can't be shipped to workers). Alternatively launch
        the script yourself with ``torchrun --nproc_per_node=N`` and call ``train(...)``
        without ``devices`` — each worker detects the distributed env and joins the group.

        Optimizer groups, LR schedule, EMA, AMP and grad-clip all come from this
        model's :class:`~dfine.config.DFINEConfig`. Progress is visualized like upstream
        D-FINE: a live console readout plus TensorBoard scalars and a ``loss_curve.png``
        under ``output_dir`` (and W&B if ``use_wandb``); only rank 0 writes them. Returns
        ``self``; the trained (EMA) weights replace ``self.model``.

        When a ``val_loader`` is available (passed, or auto-built from ``data``) and no
        ``val_fn`` is given, COCO metrics are computed each epoch via
        :func:`~dfine.train.evaluator.coco_val_fn` and logged alongside the loss.
        """
        from .train.distributed import launched_via_torchrun, setup_distributed

        if devices is not None and int(devices) > 1 and not launched_via_torchrun():
            return self._train_multigpu(
                int(devices),
                data=data,
                epochs=epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                augment=augment,
                remap_mscoco_category=remap_mscoco_category,
                output_dir=output_dir,
                use_wandb=use_wandb,
                visualize=visualize,
            )

        if launched_via_torchrun():
            setup_distributed()
            self._bind_local_rank_device()

        self._fit(
            train_loader=train_loader,
            epochs=epochs,
            data=data,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
            remap_mscoco_category=remap_mscoco_category,
            val_loader=val_loader,
            val_fn=val_fn,
            output_dir=output_dir,
            use_wandb=use_wandb,
            visualize=visualize,
        )
        return self

    def _fit(
        self,
        *,
        train_loader,
        epochs,
        data,
        batch_size,
        num_workers,
        augment,
        remap_mscoco_category,
        val_loader,
        val_fn,
        output_dir,
        use_wandb,
        visualize,
    ):
        """Build the loaders (if ``data=``) and run the training loop in this process."""
        if data is not None:
            if train_loader is not None:
                raise ValueError("Pass either `data=` or `train_loader=`, not both.")
            if self.config.task in ("segment", "sem_seg"):
                # YOLO-style seg root (images/ + labels/); box/COCO val eval is not wired
                # for seg yet, so no val loader is auto-built (pass one explicitly if needed).
                from .train.seg_dataset import build_seg_dataloader

                train_loader = build_seg_dataloader(
                    data, cfg=self.config, batch_size=batch_size, num_workers=num_workers
                )
            else:
                from .train.dataset import build_coco_dataloaders

                train_loader, auto_val_loader = build_coco_dataloaders(
                    data,
                    cfg=self.config,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    augment=augment,
                    remap_mscoco_category=remap_mscoco_category,
                )
                if val_loader is None:
                    val_loader = auto_val_loader
        elif train_loader is None:
            raise ValueError("Provide training data via `data=` or `train_loader=`.")

        # COCO box metrics need the detection postprocessor; sem_seg has no box eval.
        if val_loader is not None and val_fn is None and self.config.task != "sem_seg":
            from .train.evaluator import coco_val_fn

            val_fn = coco_val_fn(self.postprocessor, self.device)

        from .train import Trainer

        trainer = Trainer(
            self.model,
            self.config,
            device=self.device,
            output_dir=output_dir,
            visualize=visualize,
            use_wandb=use_wandb,
        )
        best = trainer.fit(train_loader, epochs=epochs, val_loader=val_loader, val_fn=val_fn)
        self.model = best.to(self.device)

    def _bind_local_rank_device(self) -> None:
        """Pin this process to its ``LOCAL_RANK`` GPU (torchrun path; no-op on CPU)."""
        if not torch.cuda.is_available():
            return
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        self.model.to(self.device)
        self.postprocessor.to(self.device)

    def _train_multigpu(
        self,
        world_size: int,
        *,
        data,
        epochs,
        batch_size,
        num_workers,
        augment,
        remap_mscoco_category,
        output_dir,
        use_wandb,
        visualize,
    ) -> DFINE:
        """Spawn ``world_size`` DDP workers, then load rank 0's trained weights back."""
        if data is None:
            raise ValueError(
                "Multi-GPU training (`devices>1`) needs `data=` (a COCO root); in-memory "
                "loaders can't be shipped to worker processes."
            )
        from .train.distributed import spawn

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        init_ckpt = out / "_init_weights.pth"
        torch.save(self.model.state_dict(), init_ckpt)

        worker_kwargs = dict(
            data=str(data),
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
            remap_mscoco_category=remap_mscoco_category,
            output_dir=str(output_dir),
            use_wandb=use_wandb,
            visualize=visualize,
        )
        try:
            spawn(_train_worker, world_size, args=(self.config, str(init_ckpt), worker_kwargs))
        finally:
            init_ckpt.unlink(missing_ok=True)

        self.load(str(out / "last.pth"))
        return self

    def val(
        self,
        data: str | os.PathLike | None = None,
        *,
        val_loader=None,
        batch_size: int = 4,
        num_workers: int = 4,
        remap_mscoco_category: bool = False,
    ) -> dict[str, float]:
        """Evaluate the model on a COCO val set and return the metrics dict.

        Provide the data one of two ways (mutually exclusive):

        * ``data="path/to/coco"`` — a COCO root; the val loader is built from
          ``val2017/`` + ``annotations/instances_val2017.json`` for you.
        * ``val_loader=...`` — a ready loader from ``build_coco_dataloader`` (its
          dataset must carry the ground-truth ``.coco``).

        Returns the 12 standard COCO metrics keyed by name (``AP`` is the primary
        mAP@[.50:.95]); see :data:`~dfine.train.evaluator.COCO_STAT_NAMES`. For stock
        MS-COCO ground truth (sparse category ids), build the model with
        ``remap_mscoco_category=True`` so predicted labels match the annotations.
        """
        if data is None and val_loader is None:
            raise ValueError("Provide validation data via `data=` or `val_loader=`.")
        if data is not None and val_loader is not None:
            raise ValueError("Pass either `data=` or `val_loader=`, not both.")
        if data is not None:
            from .train.dataset import build_coco_val_dataloader

            val_loader = build_coco_val_dataloader(
                data,
                cfg=self.config,
                batch_size=batch_size,
                num_workers=num_workers,
                remap_mscoco_category=remap_mscoco_category,
            )

        from .train.evaluator import evaluate

        return evaluate(self.model, self.postprocessor, val_loader, self.device)

    def export(
        self,
        format: str = "onnx",
        file: str | os.PathLike | None = None,
        *,
        imgsz: int | None = None,
        batch: int = 1,
        dynamic: bool = True,
        simplify: bool = False,
        opset: int = 16,
    ) -> Path:
        """Export the model to a deployable graph (Phase 3).

        Currently ``format="onnx"``: writes a single ONNX graph, batch dim dynamic by
        default. The outputs follow the model's ``task``:

        - ``detect``  — ``(images, orig_target_sizes)`` → ``(labels, boxes, scores)``.
        - ``segment`` — same inputs → ``(labels, boxes, scores, masks)`` (masks are the
          top-k queries' sigmoid maps at 1/4 res; threshold/resize/clip on the host).
        - ``sem_seg`` — ``images`` → ``sem_seg`` ``[N, H, W]`` uint8 label map (argmax
          fused in; resize to the original size on the host).

        Returns the output :class:`~pathlib.Path`. Needs ``pip install pydfine[export]``.
        ``file`` defaults to ``dfine-<size>.onnx``. Use ``simplify=True`` for ``onnxsim``,
        and :func:`dfine.export.tensorrt_command` for a downstream ``trtexec`` engine.
        """
        if format != "onnx":
            raise ValueError(f"Unsupported export format {format!r}; only 'onnx' is available.")
        from .export.onnx import export_onnx

        imgsz = imgsz or self.config.imgsz
        if imgsz != self.config.imgsz:
            raise ValueError(
                f"export imgsz={imgsz} must match the model's imgsz={self.config.imgsz}; "
                f"rebuild the model with DFINE(size=..., imgsz={imgsz}) to export at that size."
            )
        file = (
            Path(file) if file is not None else Path(f"dfine-{self.config.size or 'custom'}.onnx")
        )
        return export_onnx(
            self.model,
            self.postprocessor,
            file,
            task=self.config.task,
            imgsz=imgsz,
            batch=batch,
            opset=opset,
            dynamic=dynamic,
            simplify=simplify,
            device=self.device,
        )

    @staticmethod
    def _make_tracker(frame_rate: float):
        """Build a fresh ByteTrack tracker (raises a clear error if scipy is missing)."""
        try:
            from .track import ByteTrack
        except ImportError as e:  # pragma: no cover - trivial guard
            raise ImportError(
                "track=True needs scipy — install it with `pip install scipy` or "
                "`pip install pydfine[track]`."
            ) from e
        return ByteTrack(frame_rate=frame_rate)

    def _iter_video(self, source, conf: float, imgsz: int | None, track: bool = False):
        """Yield one :class:`Results` per decoded frame (frames read as RGB)."""
        cv2 = _require_cv2()
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {source!r}")
        tracker = self._make_tracker(cap.get(cv2.CAP_PROP_FPS) or 30.0) if track else None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.predict(rgb, conf=conf, imgsz=imgsz)[0]
                yield tracker.update(result) if tracker is not None else result
        finally:
            cap.release()

    def predict_video(
        self,
        source,
        output: str | os.PathLike = "output.mp4",
        conf: float = 0.25,
        imgsz: int | None = None,
        stream: bool = False,
        track: bool = False,
    ):
        """Detect objects frame-by-frame in a video.

        With ``stream=True`` returns a generator of per-frame :class:`Results` and
        writes nothing. Otherwise writes an annotated video to ``output`` (original
        resolution/fps) and returns its :class:`~pathlib.Path`.

        With ``track=True`` each frame's detections are run through a ByteTrack tracker
        so boxes carry a persistent ``boxes.id`` across frames (rendered as ``#id`` and
        colored per track). Needs scipy (the ``[track]`` extra).
        """
        if stream:
            return self._iter_video(source, conf, imgsz, track)

        cv2 = _require_cv2()
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {source!r}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        tracker = self._make_tracker(fps) if track else None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.predict(rgb, conf=conf, imgsz=imgsz)[0]
                if tracker is not None:
                    result = tracker.update(result)
                writer.write(cv2.cvtColor(result.plot(), cv2.COLOR_RGB2BGR))
        finally:
            cap.release()
            writer.release()
        return Path(output)

    def __repr__(self) -> str:
        size = self.config.size or "custom"
        return (
            f"DFINE(size={size!r}, num_classes={self.config.num_classes}, device='{self.device}')"
        )
