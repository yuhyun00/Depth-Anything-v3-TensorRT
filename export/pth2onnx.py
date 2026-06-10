"""
Export a Depth-Anything-3 checkpoint to ONNX (single-view / monocular).

Input ``images`` is (1, 3, H, W), ImageNet-normalized, H/W divisible by 14. The
graph returns the raw network predictions; post-processing (sky masking, metric
scaling, nested alignment) runs in Python at inference (see ../depth_estimation.py).
Exporting a single view (S = 1) skips the multi-view path (gated on S >= 3) and
keeps the model traceable.

Output tensors per family:
    anyview (small/base/large/giant):  depth, conf, intrinsics
    metric  (da3metric-large):         depth, sky
    mono    (da3mono-large):           depth, sky
    nested  (da3nested-giant-large):   depth, conf, intrinsics, metric_depth, sky

``depth_anything_3`` must be importable. Pass ``-m`` a local snapshot dir
(config.json + model.safetensors) or a HF repo id (e.g. depth-anything/DA3METRIC-LARGE).

Models over the 2GB protobuf limit (giant / nested) write weights to a sidecar
(``<name>.onnx_data``) next to the ``.onnx``; keep the two together.

Example:
    python pth2onnx.py -m depth-anything/DA3METRIC-LARGE -o da3metric_large.onnx \
        --res 504 --check
"""

import argparse
import math
import os

import torch
import torch.nn as nn

from depth_anything_3.api import DepthAnything3


# ONNX export does not support aten::cartesian_prod, so replace it with an
# equivalent built from meshgrid + stack + reshape (all ONNX-supported).
def _cartesian_prod(*tensors: torch.Tensor) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    grids = torch.meshgrid(*tensors, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1)


torch.cartesian_prod = _cartesian_prod


# The DINOv2 backbone bakes its position encodings at the trace resolution, so the
# engine would reject any other input size. Re-express both so they follow the
# dynamic input H/W at run time.
def _patch_dynamic_pos_encoding():
    from depth_anything_3.model.dinov2.layers.rope import PositionGetter
    from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer

    def interpolate_pos_encoding(self, x, w, h):
        # Interpolate pos_embed to the actual patch grid via `size` (from the
        # dynamic H/W), not the constant scale_factor path that froze the size.
        previous_dtype = x.dtype
        N = self.pos_embed.shape[1] - 1
        M = int(math.sqrt(N))  # fixed parameter -> safe to read as a constant
        dim = x.shape[-1]
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:].reshape(1, M, M, dim).permute(0, 3, 1, 2)
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(w0, h0), mode="bicubic",
            antialias=self.interpolate_antialias,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def position_getter_call(self, batch_size, height, width, device):
        # Recompute every call (skip the size-keyed cache) so the grid tracks the
        # dynamic patch count instead of freezing at the trace size.
        y_coords = torch.arange(height, device=device)
        x_coords = torch.arange(width, device=device)
        positions = torch.cartesian_prod(y_coords, x_coords)
        return positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

    DinoVisionTransformer.interpolate_pos_encoding = interpolate_pos_encoding
    PositionGetter.__call__ = position_getter_call


_patch_dynamic_pos_encoding()


# Per-family output tensor names. Keep in sync with depth_estimation.py.
OUTPUT_NAMES = {
    "anyview": ["depth", "conf", "intrinsics"],
    "metric": ["depth", "sky"],
    "mono": ["depth", "sky"],
    "nested": ["depth", "conf", "intrinsics", "metric_depth", "sky"],
}


def needs_external_data(model_name: str, model_type: str) -> bool:
    """True for giant-backbone models (anyview-giant, nested) that exceed 2GB."""
    return model_type == "nested" or "giant" in model_name.lower()


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


# Single-view export wrappers: call the network components directly, skipping the
# non-exportable Python post-processing in the original forward().
def _run_single_net(net, x: torch.Tensor):
    """Run one DepthAnything3Net (backbone + head) for S = 1.

    Returns the raw head output and per-stage features (feed feats[-1][1] to cam_dec).
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
    """da3nested-giant-large: raw tensors from both branches (scaling/alignment/sky
    handling are done in Python at inference time)."""

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
    return wrapper, model_type, da3.model_name


def main(args):
    model, model_type, model_name = build_export_model(args)

    # H/W are dynamic, so --res only sizes the trace sample; the real range is set
    # by onnx2trt's optimization profile.
    assert args.res % 14 == 0, "res must be a multiple of 14"
    data = torch.rand(1, 3, args.res, args.res)

    with torch.no_grad():
        _ = model(data)  # sanity forward before tracing

    output_names = OUTPUT_NAMES[model_type]
    dynamic_axes = {"images": {2: "H", 3: "W"}}
    for name in output_names:
        if name == "intrinsics":
            continue  # fixed (1, 3, 3)
        dynamic_axes[name] = {1: "H", 2: "W"}

    output_file = args.output or (args.model.rstrip("/").split("/")[-1] + ".onnx")

    large = needs_external_data(model_name, model_type)

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

    if large:
        import onnx

        # Move weights to a single sidecar next to the .onnx (keeps the proto < 2GB).
        data_file = os.path.basename(output_file) + "_data"
        onnx_model = onnx.load(output_file)
        onnx.save(
            onnx_model,
            output_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file,
            size_threshold=1024,
        )
        print(f"Exported ONNX -> {output_file} (+ external data {data_file})  (outputs: {output_names})")
    else:
        print(f"Exported ONNX -> {output_file}  (outputs: {output_names})")

    if args.check:
        import onnx

        # Pass the path (not a loaded model) to handle external data / >2GB.
        onnx.checker.check_model(output_file)
        print("ONNX check passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Depth-Anything-3 to ONNX (monocular)")
    parser.add_argument("-m", "--model", required=True,
                        help="HF repo id or local snapshot dir (config.json + model.safetensors)")
    parser.add_argument("-o", "--output", default=None, help="output .onnx path")
    parser.add_argument("-mt", "--model-type", default=None,
                        choices=["anyview", "metric", "mono", "nested"],
                        help="override the auto-detected model family")
    parser.add_argument("--res", type=int, default=504,
                        help="square trace-sample size (multiple of 14); ONNX H/W stay "
                             "dynamic so this does not constrain the exported model")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--check", action="store_true", help="validate the exported model")
    main(parser.parse_args())
