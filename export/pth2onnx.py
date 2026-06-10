"""
Export a Depth-Anything-3 checkpoint to ONNX (single-view / monocular).

This follows the official Depth-Anything-3 repository
(https://github.com/ByteDance-Seed/Depth-Anything-3). The exported graph takes a
single, already pre-processed image tensor ``images`` of shape ``(1, 3, H, W)``
(ImageNet-normalized, H/W divisible by 14) and returns the *raw* per-pixel network
predictions. Everything that cannot be expressed cleanly as a static graph
(sky masking with quantiles, the nested model's least-squares alignment, and the
metric ``focal * raw / 300`` scaling) is intentionally left out of the engine and
performed in Python during inference (see ../depth_estimation.py).

A single-view export (S = 1) is the key simplification that makes the model
traceable: the multi-view "reference-view selection" inside the DINO backbone is
gated on ``S >= 3`` and is therefore skipped.

Output tensors depend on the model family:
    anyview (small/base/large/giant):  depth, conf, intrinsics
    metric  (da3metric-large):         depth, sky
    mono    (da3mono-large):           depth, sky
    nested  (da3nested-giant-large):   depth, conf, intrinsics, metric_depth, sky

IMPORTANT: run this from *inside the Depth-Anything-3 repository* (or with its
``src`` on PYTHONPATH) so that ``depth_anything_3`` is importable. Point ``-m`` at
a local snapshot directory of the model (the folder that holds ``config.json`` and
``model.safetensors``), or at a Hugging Face repo id such as
``depth-anything/DA3METRIC-LARGE``.

Example:
    python pth2onnx.py -m depth-anything/DA3METRIC-LARGE -o da3metric_large.onnx \
        --height 504 --width 504 --check --simplify
"""

import argparse

import torch
import torch.nn as nn

from depth_anything_3.api import DepthAnything3


# --------------------------------------------------------------------------- #
# Per-family output tensor names. Keep in sync with depth_estimation.py.
# --------------------------------------------------------------------------- #
OUTPUT_NAMES = {
    "anyview": ["depth", "conf", "intrinsics"],
    "metric": ["depth", "sky"],
    "mono": ["depth", "sky"],
    "nested": ["depth", "conf", "intrinsics", "metric_depth", "sky"],
}


def infer_model_type(model_name: str) -> str:
    """Map a Depth-Anything-3 preset/repo name to one of our four families."""
    name = model_name.lower()
    if "nested" in name:
        return "nested"
    if "metric" in name:
        return "metric"
    if "mono" in name:
        return "mono"
    return "anyview"


def _intrinsics_from_pose_enc(pose_enc: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Build a (B, S, 3, 3) intrinsics matrix from the camera-decoder pose encoding.

    Mirrors ``pose_encoding_to_extri_intri`` (depth_anything_3.model.utils.transform)
    but only the intrinsics part, so we avoid the quaternion->matrix branch that we
    do not need for monocular depth.
    pose_enc layout: [tx, ty, tz, qx, qy, qz, qw, fov_h, fov_w].
    """
    fov_h = pose_enc[..., 7]
    fov_w = pose_enc[..., 8]
    fy = (H / 2.0) / torch.clamp(torch.tan(fov_h / 2.0), min=1e-6)
    fx = (W / 2.0) / torch.clamp(torch.tan(fov_w / 2.0), min=1e-6)

    B, S = pose_enc.shape[:2]
    intr = torch.zeros(B, S, 3, 3, dtype=pose_enc.dtype, device=pose_enc.device)
    intr[..., 0, 0] = fx
    intr[..., 1, 1] = fy
    intr[..., 0, 2] = W / 2.0
    intr[..., 1, 2] = H / 2.0
    intr[..., 2, 2] = 1.0
    return intr


# --------------------------------------------------------------------------- #
# Single-view export wrappers. They call the network's components directly,
# bypassing the non-exportable Python post-processing in the original forward().
# --------------------------------------------------------------------------- #
def _run_single_net(net, x: torch.Tensor):
    """Run one DepthAnything3Net (backbone + head [+ cam_dec]) for S = 1.

    Returns the raw head output Dict and the per-stage features (so callers can
    feed ``feats[-1][1]`` to the camera decoder).
    """
    H, W = x.shape[-2], x.shape[-1]
    feats, _aux = net.backbone(
        x,
        cam_token=None,
        export_feat_layers=[],
        ref_view_strategy="first",
    )
    out = net.head(feats, H, W, patch_start_idx=0)
    return out, feats, H, W


class AnyViewExport(nn.Module):
    """small / base / large / giant: depth + confidence + predicted intrinsics."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, images: torch.Tensor):
        x = images.unsqueeze(1)  # (1, 3, H, W) -> (B=1, S=1, 3, H, W)
        out, feats, H, W = _run_single_net(self.net, x)

        depth = out["depth"][0]  # (S, H, W) -> (1, H, W)
        conf = out["depth_conf"][0]  # (1, H, W)

        pose_enc = self.net.cam_dec(feats[-1][1])  # (B, S, 9)
        intrinsics = _intrinsics_from_pose_enc(pose_enc, H, W)[0]  # (1, 3, 3)
        return depth, conf, intrinsics


class DPTExport(nn.Module):
    """da3metric-large / da3mono-large: raw depth + sky logits."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, images: torch.Tensor):
        x = images.unsqueeze(1)
        out, _feats, _H, _W = _run_single_net(self.net, x)
        depth = out["depth"][0]  # (1, H, W)
        sky = out["sky"][0]  # (1, H, W)
        return depth, sky


class NestedExport(nn.Module):
    """da3nested-giant-large: raw tensors from both branches.

    The metric scaling (focal/300), least-squares alignment to the any-view depth
    and the sky handling are all done in Python at inference time.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net  # NestedDepthAnything3Net

    def forward(self, images: torch.Tensor):
        x = images.unsqueeze(1)

        # Any-view (giant) branch -> relative depth, confidence, intrinsics.
        g_out, g_feats, H, W = _run_single_net(self.net.da3, x)
        depth = g_out["depth"][0]  # (1, H, W)
        conf = g_out["depth_conf"][0]  # (1, H, W)
        g_pose = self.net.da3.cam_dec(g_feats[-1][1])
        intrinsics = _intrinsics_from_pose_enc(g_pose, H, W)[0]  # (1, 3, 3)

        # Metric branch -> raw metric depth + sky.
        m_out, _m_feats, _H, _W = _run_single_net(self.net.da3_metric, x)
        metric_depth = m_out["depth"][0]  # (1, H, W)
        sky = m_out["sky"][0]  # (1, H, W)

        return depth, conf, intrinsics, metric_depth, sky


WRAPPERS = {
    "anyview": AnyViewExport,
    "metric": DPTExport,
    "mono": DPTExport,
    "nested": NestedExport,
}


def build_export_model(args):
    da3 = DepthAnything3.from_pretrained(args.model)
    da3.eval()
    model_type = args.model_type or infer_model_type(da3.model_name)
    print(f"Loaded '{da3.model_name}' -> family '{model_type}'")

    wrapper = WRAPPERS[model_type](da3.model).eval()
    return wrapper, model_type


def main(args):
    model, model_type = build_export_model(args)

    # H/W must be divisible by 14 (DINO patch size).
    assert args.height % 14 == 0 and args.width % 14 == 0, "height/width must be multiples of 14"
    data = torch.rand(1, 3, args.height, args.width)

    with torch.no_grad():
        _ = model(data)  # sanity forward before tracing

    output_names = OUTPUT_NAMES[model_type]
    # H and W are dynamic so the engine can accept aspect-ratio-preserving inputs.
    dynamic_axes = {"images": {2: "H", 3: "W"}}
    for name in output_names:
        if name == "intrinsics":
            continue  # fixed (1, 3, 3)
        dynamic_axes[name] = {1: "H", 2: "W"}

    output_file = args.output or (args.model.rstrip("/").split("/")[-1] + ".onnx")

    torch.onnx.export(
        model,
        (data,),
        output_file,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"Exported ONNX -> {output_file}  (outputs: {output_names})")

    if args.check:
        import onnx

        onnx.checker.check_model(onnx.load(output_file))
        print("ONNX check passed.")

    if args.simplify:
        import onnx
        import onnxsim

        onnx_model = onnx.load(output_file)
        onnx_model, ok = onnxsim.simplify(onnx_model)
        assert ok, "onnxsim simplification failed"
        onnx.save(onnx_model, output_file)
        print(f"Simplified ONNX -> {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Depth-Anything-3 to ONNX (monocular)")
    parser.add_argument("-m", "--model", required=True,
                        help="HF repo id or local snapshot dir (config.json + model.safetensors)")
    parser.add_argument("-o", "--output", default=None, help="output .onnx path")
    parser.add_argument("-mt", "--model-type", default=None,
                        choices=["anyview", "metric", "mono", "nested"],
                        help="override the auto-detected model family")
    parser.add_argument("--height", type=int, default=504, help="sample (opt) height, multiple of 14")
    parser.add_argument("--width", type=int, default=504, help="sample (opt) width, multiple of 14")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--check", action="store_true", help="validate the exported model")
    parser.add_argument("--simplify", action="store_true", help="simplify with onnxsim")
    main(parser.parse_args())
