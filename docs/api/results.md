# Results & Boxes

`DFINE.predict` returns a list of `Results` (one per image); each holds the detected
`Boxes` (original-scale `xyxy`), with `.plot()`/`.save()` and interop converters
(`to_pandas`/`to_coco`/`to_supervision`).

::: dfine.results.Results

::: dfine.results.Boxes
