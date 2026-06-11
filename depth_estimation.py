"""
Depth-Anything-3 (TensorRT) monocular depth-estimation pipeline.

Pipeline: RGB -> preprocess -> TensorRT inference -> family-specific
post-processing (sky handling, metric scaling, nested alignment) -> depth (+ conf).

Model families (must match the exported engine):
    anyview  (DA3-SMALL/BASE/LARGE/GIANT)  -> relative (affine-invariant) depth
    metric   (DA3METRIC-LARGE)             -> metric depth (needs --focal)
    mono     (DA3MONO-LARGE)               -> relative monocular depth
    nested   (DA3NESTED-GIANT-LARGE)       -> metric depth (self-contained)

The input size is read from the engine (longest side of its opt shape, set by
``onnx2trt --res``), so there is no input-size flag: each image is resized with the
longest side at that value, aspect preserved, each side a multiple of 14.

Example:
    python3 depth_estimation.py \
        --input ./images --output ./results \
        --trt ./da3metric_large.engine --model-type metric \
        --focal 1200 --save-raw
"""

import argparse
import collections
import glob
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision.transforms as T

import tensorrt as trt


# --------------------------------------------------------------------------- #
# TensorRT inference wrapper (dynamic spatial shapes).
# --------------------------------------------------------------------------- #
class TRTInference(object):
    """Runs a Depth-Anything-3 engine with a single dynamic ``images`` input.

    Output tensors are (re)allocated every call from the context's resolved
    shapes, which is required because H/W change from image to image.
    """

    def __init__(self, engine_path, device="cuda:0", verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        trt.init_libnvinfer_plugins(self.logger, "")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names = self._names(trt.TensorIOMode.INPUT)
        self.output_names = self._names(trt.TensorIOMode.OUTPUT)
        assert len(self.input_names) == 1, "expected a single image input"

        # Accepted input-size range (min/max) and tuned size (opt) from the engine.
        mn, opt, mx = self.engine.get_tensor_profile_shape(self.input_names[0], 0)
        self.min_shape = tuple(int(d) for d in mn)  # (1, 3, Hmin, Wmin)
        self.opt_shape = tuple(int(d) for d in opt)  # (1, 3, Hopt, Wopt)
        self.max_shape = tuple(int(d) for d in mx)  # (1, 3, Hmax, Wmax)

    def _names(self, mode):
        names = []
        for name in self.engine:
            if self.engine.get_tensor_mode(name) == mode:
                names.append(name)
        return names

    def __call__(self, images: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        """images: float32 CUDA tensor of shape (1, 3, H, W)."""
        images = images.to(self.device).contiguous()
        in_name = self.input_names[0]
        self.context.set_input_shape(in_name, tuple(images.shape))
        self.context.set_tensor_address(in_name, images.data_ptr())

        outputs = OrderedDict()
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            buf = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            outputs[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())

        stream = torch.cuda.current_stream(device=self.device)
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outputs


# --------------------------------------------------------------------------- #
# Depth-Anything-3 pipeline.
# --------------------------------------------------------------------------- #
PATCH_SIZE = 14
_METRIC_SCALE = 300.0  # metric_depth = focal * raw / 300


class DepthAnythingV3:
    """End-to-end Depth-Anything-3 (TensorRT) monocular depth estimator.

    ``run()`` returns a dict:
        depth      : (H0, W0) float32 depth map, resized back to the original image
        conf       : (H0, W0) float32 confidence map, or None
        is_metric  : bool, True when ``depth`` is in metres
    """

    def __init__(self, engine_path, model_type,
                 process_res_method="upper_bound_resize", device="cuda:0",
                 focal=None):
        assert model_type in {"anyview", "metric", "mono", "nested"}
        assert process_res_method in {"upper_bound_resize", "lower_bound_resize"}
        self.model_type = model_type
        self.process_res_method = process_res_method
        self.device = device
        self.focal = focal  # focal length in pixels of the ORIGINAL image (metric model)

        self.model = TRTInference(engine_path, device=device)

        # Input size comes from the engine: resize targets the longest opt side,
        # clamped to the [min, max] range. Nothing is passed in by the caller.
        self.min_h, self.min_w = self.model.min_shape[2], self.model.min_shape[3]
        self.opt_h, self.opt_w = self.model.opt_shape[2], self.model.opt_shape[3]
        self.max_h, self.max_w = self.model.max_shape[2], self.model.max_shape[3]
        self.process_res = max(self.opt_h, self.opt_w)
        print(f"  process_res from engine ({self.opt_h}x{self.opt_w}) -> {self.process_res}")
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ---- 1) preprocessing -------------------------------------------------- #
    def preprocess(self, rgb_image):
        """RGB uint8 (H0,W0,3) -> ((1,3,H,W) blob, meta dict)."""
        orig_h, orig_w = rgb_image.shape[:2]

        # (a) boundary resize, preserving aspect ratio
        img = self._boundary_resize(rgb_image, self.process_res, self.process_res_method)
        # (b) round each side to a multiple of PATCH_SIZE
        img = self._make_divisible_by_resize(img, PATCH_SIZE)
        # (c) keep within the engine's [min, max] range
        img = self._fit_engine_bounds(img)
        proc_h, proc_w = img.shape[:2]

        blob = self.normalize(img)[None]  # (1, 3, H, W), float32

        # focal scales by the same single (longest-side) resize factor.
        scale = max(proc_h, proc_w) / float(max(orig_h, orig_w))
        meta = {
            "orig_hw": (orig_h, orig_w),
            "proc_hw": (proc_h, proc_w),
            "focal_scale": scale,
        }
        return blob.to(self.device), meta

    @staticmethod
    def _boundary_resize(img, target, method):
        h, w = img.shape[:2]
        ref = max(w, h) if method == "upper_bound_resize" else min(w, h)
        if ref == target:
            return img
        s = target / float(ref)
        new_w = max(1, int(round(w * s)))
        new_h = max(1, int(round(h * s)))
        interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    @staticmethod
    def _make_divisible_by_resize(img, patch):
        h, w = img.shape[:2]

        def nearest_multiple(x, p):
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        new_w = max(patch, nearest_multiple(w, patch))
        new_h = max(patch, nearest_multiple(h, patch))
        if new_w == w and new_h == h:
            return img
        interp = cv2.INTER_CUBIC if (new_w > w or new_h > h) else cv2.INTER_AREA
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    def _fit_engine_bounds(self, img):
        """Clamp H/W into the engine's [min, max] range. Normally a no-op; only fires
        for extreme aspect ratios. Bounds are multiples of 14, so output stays so."""
        h, w = img.shape[:2]
        new_h = min(max(h, self.min_h), self.max_h)
        new_w = min(max(w, self.min_w), self.max_w)
        if (new_h, new_w) == (h, w):
            return img
        print(f"  [warn] processed size ({h},{w}) outside engine range "
              f"H[{self.min_h},{self.max_h}] W[{self.min_w},{self.max_w}]; "
              f"resizing to ({new_h},{new_w}).")
        interp = cv2.INTER_CUBIC if (new_h > h or new_w > w) else cv2.INTER_AREA
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    # ---- 2) TensorRT inference -------------------------------------------- #
    def infer(self, blob):
        out = self.model(blob)
        # Squeeze the leading view dim (N=1) -> (H, W) numpy arrays where present.
        np_out = {}
        for k, v in out.items():
            arr = v.detach().float().cpu().numpy()
            if k == "intrinsics":
                np_out[k] = arr.reshape(3, 3)
            else:
                np_out[k] = arr[0]  # (1,H,W) -> (H,W)
        return np_out

    # ---- 3) post-processing ----------------------------------------------- #
    def postprocess(self, out, meta):
        if self.model_type == "anyview":
            depth, conf, is_metric = out["depth"], out.get("conf"), False
        elif self.model_type == "mono":
            depth = self._apply_sky(out["depth"], out["sky"])
            conf, is_metric = None, False
        elif self.model_type == "metric":
            depth, conf, is_metric = self._postprocess_metric(out, meta)
        else:  # nested
            depth, conf, is_metric = self._postprocess_nested(out)

        # Resize prediction back to the original image resolution.
        orig_h, orig_w = meta["orig_hw"]
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        if conf is not None:
            conf = cv2.resize(conf, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Intrinsics (when the model outputs them) rescaled to the original
        # resolution, so K pairs with the original-resolution depth map.
        K = out.get("intrinsics")
        if K is not None:
            proc_h, proc_w = meta["proc_hw"]
            sx, sy = orig_w / float(proc_w), orig_h / float(proc_h)
            K = np.asarray(K, dtype=np.float64).copy()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy

        return {"depth": depth.astype(np.float32),
                "conf": None if conf is None else conf.astype(np.float32),
                "is_metric": is_metric,
                "K": K}

    def _postprocess_metric(self, out, meta):
        """Standalone metric model: metric_depth = focal * raw / 300, then sky fill."""
        raw = out["depth"]
        if self.focal is None:
            print("  [warn] --focal not provided for a metric model; returning RELATIVE depth.")
            depth = self._apply_sky(raw, out["sky"])
            return depth, None, False
        focal_proc = float(self.focal) * meta["focal_scale"]  # focal at processed resolution
        depth = raw * (focal_proc / _METRIC_SCALE)
        depth = self._apply_sky(depth, out["sky"])
        return depth, None, True

    def _postprocess_nested(self, out):
        """Nested model: scale metric branch by focal/300, least-squares align to the
        any-view (giant) depth, then fill the sky."""
        depth = out["depth"]          # giant relative depth (H, W)
        conf = out["conf"]            # giant confidence (H, W)
        intr = out["intrinsics"]      # (3, 3)
        metric = out["metric_depth"]  # metric branch raw depth (H, W)
        sky = out["sky"]              # metric branch sky prob (H, W)

        focal = (intr[0, 0] + intr[1, 1]) / 2.0
        metric = metric * (focal / _METRIC_SCALE)

        non_sky = sky < 0.3
        if non_sky.sum() <= 10:
            return depth, conf, False

        # alignment mask: confident, non-sky, positive depths
        conf_ns = conf[non_sky]
        median_conf = np.quantile(conf_ns, 0.5)
        align = (conf >= median_conf) & non_sky & (metric > 1e-2) & (depth > 1e-3)
        if align.sum() <= 10:
            return depth, conf, False

        a = metric[align].astype(np.float64)
        b = depth[align].astype(np.float64)
        scale = float(np.dot(a, b) / max(np.dot(b, b), 1e-12))  # metric ~= scale * depth
        depth = depth * scale  # giant depth -> metres

        # Sky -> 99th percentile of non-sky metric depth, capped at 200 m.
        ns_depth = depth[non_sky]
        non_sky_max = min(float(np.quantile(ns_depth, 0.99)), 200.0)
        depth = depth.copy()
        depth[~non_sky] = non_sky_max
        conf = conf.copy()
        conf[~non_sky] = 1.0
        return depth, conf, True

    @staticmethod
    def _apply_sky(depth, sky, sky_threshold=0.3, max_cap=None):
        """Set sky pixels to the 99th percentile of non-sky depth."""
        non_sky = sky < sky_threshold
        if non_sky.sum() <= 10 or (~non_sky).sum() <= 10:
            return depth
        non_sky_max = float(np.quantile(depth[non_sky], 0.99))
        if max_cap is not None:
            non_sky_max = min(non_sky_max, max_cap)
        depth = depth.copy()
        depth[~non_sky] = non_sky_max
        return depth

    # ---- 4) everything together ------------------------------------------- #
    def run(self, rgb_image):
        blob, meta = self.preprocess(rgb_image)
        out = self.infer(blob)
        return self.postprocess(out, meta)


# --------------------------------------------------------------------------- #
# Visualization (separate from the pipeline class).
# --------------------------------------------------------------------------- #
def visualize_depth(depth, conf=None, is_metric=False, colormap=cv2.COLORMAP_INFERNO,
                    lo_pct=2.0, hi_pct=98.0):
    """Colorize a depth map. Returns a BGR uint8 image (for cv2.imwrite).

    Robust min/max via percentiles so a few outliers don't wash out the colors.
    Closer = brighter (depth is inverted before colorizing).
    """
    valid = np.isfinite(depth)
    if not valid.any():
        return np.zeros(depth.shape + (3,), dtype=np.uint8)

    lo = np.percentile(depth[valid], lo_pct)
    hi = np.percentile(depth[valid], hi_pct)
    denom = max(hi - lo, 1e-6)
    norm = np.clip((depth - lo) / denom, 0.0, 1.0)
    inv = 1.0 - norm  # near -> high value -> bright
    vis = cv2.applyColorMap((inv * 255).astype(np.uint8), colormap)
    return vis


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _collect_images(input_dir):
    files = []
    for ext in _IMG_EXTS:
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="Depth-Anything-3 (TensorRT) depth estimation")
    parser.add_argument("-i", "--input", required=True, help="folder containing input images")
    parser.add_argument("-o", "--output", required=True, help="folder to save results")
    parser.add_argument("-trt", "--trt", required=True, help="path to the TensorRT engine (.engine)")
    parser.add_argument("-mt", "--model-type", required=True,
                        choices=["anyview", "metric", "mono", "nested"],
                        help="model family (must match the exported engine)")
    parser.add_argument("--process-res-method", default="upper_bound_resize",
                        choices=["upper_bound_resize", "lower_bound_resize"],
                        help="how the longest/shortest side is matched to the engine size")
    parser.add_argument("--focal", type=float, default=None,
                        help="focal length in pixels of the ORIGINAL image (metric model only)")
    parser.add_argument("--save-raw", action="store_true",
                        help="also save the raw depth map as a .npy file")
    parser.add_argument("-d", "--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    images = _collect_images(args.input)
    if not images:
        print(f"No images found in {args.input}")
        return

    pipeline = DepthAnythingV3(
        engine_path=args.trt,
        model_type=args.model_type,
        process_res_method=args.process_res_method,
        device=args.device,
        focal=args.focal,
    )

    print(f"Found {len(images)} images. model_type={args.model_type}")
    for path in images:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"  [skip] could not read {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        result = pipeline.run(rgb)
        depth = result["depth"]

        vis = visualize_depth(depth, conf=result["conf"], is_metric=result["is_metric"])
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.output, stem + "_depth.png")
        cv2.imwrite(out_path, vis)
        if args.save_raw:
            np.save(os.path.join(args.output, stem + "_depth.npy"), depth)

        unit = "m" if result["is_metric"] else "rel"
        print(f"  {os.path.basename(path)}: depth[{unit}] "
              f"min={depth.min():.3f} max={depth.max():.3f} -> {out_path}")


if __name__ == "__main__":
    main()
