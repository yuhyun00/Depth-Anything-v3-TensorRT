# Depth-Anything-v3-TensorRT

*다른 언어로 보기: [English](README.md)*

[Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)를 TensorRT로
가속한 단안(monocular) 깊이 추정 파이프라인입니다. `export/` 폴더에서 체크포인트를
`pth → onnx → trt`로 변환하고 `depth_estimation.py`로 폴더 안의 이미지들에 대해 추론을 실행합니다.

- **깊이 추정**: Depth-Anything-3를 TensorRT 엔진으로 변환해 추론합니다.
- **Metric & 상대 깊이**: metric 모델(미터 단위 깊이)과 상대(affine-invariant) 모델을 모두 지원합니다.
- **후처리는 Python에서**: 엔진에는 순수 네트워크만 담고, metric 변환과 정적 그래프로 표현하기 어려운 단계(하늘 마스킹, nested 모델의 최소제곱 정렬)는 Python에서 처리합니다 — 공식 레포와 동일한 방식입니다.
- **시각화**: 파이프라인과 분리된 함수로 깊이 맵을 컬러로 변환합니다.

```
RGB 이미지 → 전처리 → Depth-Anything-3 TensorRT 추론 → 후처리(하늘 / metric / 정렬) → 깊이 맵 → 시각화
```

## 지원 모델

| 계열 (`--model-type`) | 체크포인트 | 엔진 출력 | 깊이 |
|---|---|---|---|
| `anyview` | `DA3-SMALL`, `DA3-BASE`, `DA3-LARGE`, `DA3-GIANT` | `depth, conf, intrinsics` | 상대 (affine-invariant) |
| `metric`  | `DA3METRIC-LARGE` | `depth, sky` | **metric (미터)** — `--focal` 필요 |
| `mono`    | `DA3MONO-LARGE` | `depth, sky` | 상대 |
| `nested`  | `DA3NESTED-GIANT-LARGE` | `depth, conf, intrinsics, metric_depth, sky` | **metric (미터)** — 자체 완결 |

`-1.1` 체크포인트(예: `DA3-GIANT-1.1`)는 구조는 동일하고 가중치만 갱신된 버전으로, 동일하게 동작합니다.

## 디렉터리 구조

```
Depth-Anything-v3-TensorRT/
├── depth_estimation.py   # 파이프라인 클래스 + 시각화 함수 + main()
├── export/
│   ├── pth2onnx.py      # Depth-Anything-3 체크포인트 -> .onnx
│   └── onnx2trt.py      # .onnx -> .engine (TensorRT)
├── requirements.txt
├── README.md
└── README_KR.md
```

## 설치

```bash
pip install -r requirements.txt
pip install tensorrt                       # CUDA/플랫폼에 맞게 별도 설치

# 내보내기(export) 단계에만 필요:
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
pip install -e Depth-Anything-3
```

## 사용법

### 1. 내보내기: pth → onnx

`depth_anything_3`가 import 되도록 **클론한 Depth-Anything-3 레포 내부**에서 실행하세요.
`-m`은 Hugging Face 레포 id 또는 로컬 스냅샷 폴더(`config.json` + `model.safetensors`)입니다.
모델 계열은 이름에서 자동으로 감지됩니다.

```bash
# metric 모델
python export/pth2onnx.py -m depth-anything/DA3METRIC-LARGE \
    -o da3metric_large.onnx --res 504 --check

# any-view (상대 깊이) 모델
python export/pth2onnx.py -m depth-anything/DA3-LARGE -o da3_large.onnx

# nested (metric) 모델
python export/pth2onnx.py -m depth-anything/DA3NESTED-GIANT-LARGE -o da3_nested.onnx
```

`--res`는 트레이싱에 쓰이는 정사각 *샘플* 크기만 정하며, 내보낸 그래프는 H/W를 동적으로 유지합니다
(DINO 위치 임베딩 보간과 2D RoPE도 동적으로 export됨). 따라서 어떤 종횡비든 받습니다. `--res`는 14의 배수여야 합니다.

> 2GB protobuf 한계를 넘는 모델(giant / nested)은 가중치를 `.onnx` 옆의 `<name>.onnx_data`
> 사이드카에 저장합니다 — 두 파일을 항상 같은 폴더에 두세요.

### 2. 변환: onnx → trt

긴 변만 `--res`로 지정하면 됩니다. 엔진은 두 변이 `[14, res]`인 모든 H×W를 받고 `(res, res)`에 맞춰 튜닝됩니다.
추론 스크립트가 이 값을 엔진에서 읽어오므로, 실행 시 입력 크기를 따로 지정하지 않습니다.

```bash
python export/onnx2trt.py --onnx da3metric_large.onnx --saveEngine da3metric_large.engine \
    --fp16 --res 504
```

### 3. 추론

입력 폴더의 모든 이미지를 읽어 깊이를 추정하고 결과를 출력 폴더에 저장합니다.

```bash
# metric 깊이 (미터) — 원본 이미지의 초점거리(focal, 픽셀 단위)를 전달
python depth_estimation.py -i ./images -o ./results \
    --trt da3metric_large.engine --model-type metric --focal 1200 --save-raw

# 상대 깊이
python depth_estimation.py -i ./images -o ./results \
    --trt da3_large.engine --model-type anyview

# nested metric 깊이 (focal 불필요; 카메라를 예측함)
python depth_estimation.py -i ./images -o ./results \
    --trt da3_nested.engine --model-type nested --save-raw
```

입력 크기 인자는 없습니다: 리사이즈 목표(긴 변)는 엔진의 optimization profile(`onnx2trt --res`로 설정)에서 읽어옵니다.

이미지당 출력물: 컬러 변환된 `*_depth.png`(가까울수록 밝음), 그리고 `--save-raw` 사용 시
원본 해상도의 실수형 깊이 맵을 담은 `*_depth.npy`(metric/nested 모델은 미터 단위).

#### 주요 인자

| 인자 | 설명 |
|------|------|
| `-i, --input` | 입력 이미지 폴더 |
| `-o, --output` | 결과 저장 폴더 |
| `-trt, --trt` | TensorRT 엔진(`.engine`) 경로 |
| `-mt, --model-type` | `anyview / metric / mono / nested` (내보낸 엔진과 일치해야 함) |
| `--process-res-method` | `upper_bound_resize`(긴 변 기준) 또는 `lower_bound_resize`(짧은 변 기준) |
| `--focal` | **원본** 이미지의 초점거리(픽셀 단위, metric 모델 전용) |
| `--save-raw` | 원본 깊이 맵을 `.npy`로도 저장 |
| `-d, --device` | 추론 디바이스 (기본 `cuda:0`) |

## Python API

`depth_estimation.py`의 `DepthAnythingV3` 클래스를 직접 사용할 수도 있습니다.

```python
import cv2
from depth_estimation import DepthAnythingV3, visualize_depth

pipe = DepthAnythingV3(
    engine_path="da3metric_large.engine",
    model_type="metric",
    device="cuda:0",
    focal=1200,            # 원본 이미지 초점거리(픽셀, metric 모델)
)

bgr = cv2.imread("frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

result = pipe.run(rgb)               # {"depth", "conf", "is_metric"}
depth = result["depth"]              # (H0, W0) float32, is_metric면 미터 단위

vis = visualize_depth(depth)         # 저장용 BGR 이미지
cv2.imwrite("result.png", vis)
```

### 메서드

- `preprocess(rgb_image)` — 리사이즈(크기는 엔진에서 읽음) + 정규화 → `((1,3,H,W) blob, meta)`
- `infer(blob)` — TensorRT 추론 → numpy 출력 텐서 dict
- `postprocess(out, meta)` — 하늘 / metric / 정렬 처리 후 원본 크기로 리사이즈
- `run(rgb_image)` — 위 단계를 한 번에 실행

## Metric 깊이에 대한 메모

- **`metric` 모델:** `metric_depth = focal · raw / 300`이며, 여기서 `focal`은 `fx, fy`의 평균(픽셀)입니다.
  `--focal`로 원본 이미지 픽셀 단위 값을 전달하면, 파이프라인이 처리(processed) 해상도에 맞춰 자동으로
  재스케일합니다. `--focal`이 없으면 metric 모델은 상대 깊이로 대체됩니다(경고 출력).
- **`nested` 모델:** 완전히 자체 완결적입니다. giant 분기가 카메라(focal)를 예측하고, metric 분기를
  `focal/300`으로 스케일한 뒤 giant 깊이에 최소제곱으로 정렬하고 하늘을 채웁니다 — 출력은 이미 미터 단위입니다.

## 전처리

긴 변을 엔진 크기(종횡비 유지, 확대 시 `INTER_CUBIC` / 축소 시 `INTER_AREA`)로 리사이즈 →
각 차원을 14의 가장 가까운 배수로 반올림 → `ToTensor` →
ImageNet 정규화(`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).

## 라이선스 / 출처

- Depth-Anything-3: https://github.com/ByteDance-Seed/Depth-Anything-3
