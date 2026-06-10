# Depth-Anything-v3-TensorRT

*Read this in other languages: [한국어](README_KR.md)*

Monocular depth estimation with [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
accelerated by TensorRT. An `export/` folder converts a checkpoint
`pth → onnx → trt`, and `depth_estimation.py` runs inference over a folder of images.

- **Depth**: Depth-Anything-3 is converted to a TensorRT engine for inference.
- **Metric & relative**: both metric models (depth in metres) and relative
  (affine-invariant) models are supported.
- **Post-processing in Python**: the engine holds only the pure network; the metric
  conversion and the steps that don't map to a static graph (sky masking, the
  nested model's least-squares alignment) run in Python — exactly as in the
  official repo.
- **Visualization**: a standalone function (separate from the pipeline class)
  colorizes the depth map.

```
RGB image → preprocess → Depth-Anything-3 TensorRT inference → post-processing (sky / metric / alignment) → depth map → visualization
```

## Supported models

| Family (`--model-type`) | Checkpoints | Engine outputs | Depth |
|---|---|---|---|
| `anyview` | `DA3-SMALL`, `DA3-BASE`, `DA3-LARGE`, `DA3-GIANT` | `depth, conf, intrinsics` | relative (affine-invariant) |
| `metric`  | `DA3METRIC-LARGE` | `depth, sky` | **metric (metres)** — needs `--focal` |
| `mono`    | `DA3MONO-LARGE` | `depth, sky` | relative |
| `nested`  | `DA3NESTED-GIANT-LARGE` | `depth, conf, intrinsics, metric_depth, sky` | **metric (metres)** — self-contained |

The `-1.1` checkpoints (e.g. `DA3-GIANT-1.1`) are the same architectures with
refreshed weights and work identically.

## Directory layout

```
Depth-Anything-v3-TensorRT/
├── depth_estimation.py   # pipeline class + visualization function + main()
├── export/
│   ├── pth2onnx.py      # Depth-Anything-3 checkpoint -> .onnx
│   └── onnx2trt.py      # .onnx -> .engine (TensorRT)
├── requirements.txt
├── README.md
└── README_KR.md
```

## Installation

```bash
pip install -r requirements.txt
pip install tensorrt                       # match your CUDA / platform

# needed only for the export step:
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
pip install -e Depth-Anything-3
```

## Usage

### 1. Export: pth → onnx

Run from inside the cloned Depth-Anything-3 repo (so `depth_anything_3` imports).
`-m` is a Hugging Face repo id or a local snapshot dir (`config.json` +
`model.safetensors`). The family is auto-detected from the name.

```bash
# metric model
python export/pth2onnx.py -m depth-anything/DA3METRIC-LARGE \
    -o da3metric_large.onnx --height 504 --width 504 --check --simplify

# any-view (relative) model
python export/pth2onnx.py -m depth-anything/DA3-LARGE -o da3_large.onnx --simplify

# nested (metric) model
python export/pth2onnx.py -m depth-anything/DA3NESTED-GIANT-LARGE -o da3_nested.onnx --simplify
```

`--height/--width` only set the *sample/opt* size used during tracing; the engine
keeps H and W dynamic. Both must be multiples of 14.

> Dynamic-resolution ViT export exercises the DINO positional-embedding
> interpolation and 2D RoPE. It traces cleanly with a recent PyTorch (≥ 2.1) at
> `opset 17`. If a particular TensorRT/PyTorch combination struggles with dynamic
> spatial dims, export and build at a single fixed resolution instead — set
> `--height/--width` to your target and use equal `--min/--opt/--max` in step 2 —
> and keep `--process-res` at the longest side of that resolution during inference.

### 2. Convert: onnx → trt

The input has dynamic spatial dims, so pick a `--opt` size matching what you run
most (usually `process_res × process_res`). All sizes must be multiples of 14.

```bash
python export/onnx2trt.py --onnx da3metric_large.onnx --saveEngine da3metric_large.engine \
    --fp16 --min 154 154 --opt 504 504 --max 504 504
```

### 3. Inference

Reads every image in the input folder, estimates depth, and saves the results to
the output folder.

```bash
# metric depth (metres) — pass the ORIGINAL image focal length in pixels
python depth_estimation.py -i ./images -o ./results \
    --trt da3metric_large.engine --model-type metric --focal 1200 --save-raw

# relative depth
python depth_estimation.py -i ./images -o ./results \
    --trt da3_large.engine --model-type anyview

# nested metric depth (no focal needed; camera is predicted)
python depth_estimation.py -i ./images -o ./results \
    --trt da3_nested.engine --model-type nested --save-raw
```

Outputs per image: a colorized `*_depth.png` (closer = brighter), and, with
`--save-raw`, a `*_depth.npy` holding the float depth map at the original
resolution (in metres for metric/nested models).

#### Main arguments

| Argument | Description |
|----------|-------------|
| `-i, --input` | input image folder |
| `-o, --output` | folder to save results |
| `-trt, --trt` | path to the TensorRT engine (`.engine`) |
| `-mt, --model-type` | `anyview / metric / mono / nested` (must match the exported engine) |
| `-pr, --process-res` | resize the longest side to this (official default 504) |
| `--process-res-method` | `upper_bound_resize` (longest side) or `lower_bound_resize` (shortest side) |
| `--focal` | focal length in pixels of the **original** image (metric model only) |
| `--save-raw` | also save the raw depth map as a `.npy` file |
| `-d, --device` | inference device (default `cuda:0`) |

## Python API

You can also use the `DepthAnythingV3` class from `depth_estimation.py` directly.

```python
import cv2
from depth_estimation import DepthAnythingV3, visualize_depth

pipe = DepthAnythingV3(
    engine_path="da3metric_large.engine",
    model_type="metric",
    process_res=504,
    device="cuda:0",
    focal=1200,            # original-image focal in pixels (metric model)
)

bgr = cv2.imread("frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

result = pipe.run(rgb)               # {"depth", "conf", "is_metric"}
depth = result["depth"]              # (H0, W0) float32, metres if is_metric

vis = visualize_depth(depth)         # BGR image, ready to save
cv2.imwrite("result.png", vis)
```

### Methods

- `preprocess(rgb_image)` — official `InputProcessor` replica → `((1,3,H,W) blob, meta)`
- `infer(blob)` — TensorRT inference → dict of numpy output tensors
- `postprocess(out, meta)` — sky / metric / alignment, resize back to the original
- `run(rgb_image)` — run all of the above at once

## Notes on metric depth

- **`metric` model:** `metric_depth = focal · raw / 300`, where `focal` is the
  average of `fx, fy` in pixels. Pass it via `--focal` (original-image pixels); the
  pipeline rescales it to the processed resolution automatically. Without
  `--focal`, the metric model falls back to relative depth (with a warning).
- **`nested` model:** fully self-contained. The giant branch predicts the camera
  (focal), the metric branch is scaled by `focal/300`, least-squares aligned to the
  giant depth, and the sky is filled — output is already in metres.

## Preprocessing

Replicates the official `InputProcessor`: resize the longest side to
`--process-res` (default 504, aspect preserved, `INTER_CUBIC` up / `INTER_AREA`
down), round each dimension to the nearest multiple of 14, `ToTensor`, then
ImageNet normalization (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).

## License / Sources

- Depth-Anything-3: https://github.com/ByteDance-Seed/Depth-Anything-3
