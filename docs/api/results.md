# Results & Boxes

`DFINE.predict` returns a list of `Results` (one per image); each holds the detected
`Boxes` (original-scale `xyxy`), with `.plot()`/`.save()` and interop converters
(`to_pandas`/`to_coco`/`to_supervision`).

For segmentation models a result also carries a segmentation container, both at the
**original image resolution**:

- `task="segment"` → `Results.masks` — a [`Masks`](#dfine.results.Masks) object whose
  rows align 1:1 with `Boxes` (`.data` is `[N, H, W]` bool). `Results.plot()` overlays
  each instance's mask under its box, and `to_supervision()` attaches the masks.
- `task="sem_seg"` → `Results.sem_seg` — a [`SemSeg`](#dfine.results.SemSeg) object with
  a dense uint8 `[H, W]` class-id map (`255` = void). `plot()` tints each class; these
  results are boxless (`Boxes` is empty).

On a detection result both are `None`.

::: dfine.results.Results

::: dfine.results.Boxes

::: dfine.results.Masks

::: dfine.results.SemSeg
