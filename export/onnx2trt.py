"""
Build a TensorRT engine (.engine) from a Depth-Anything-3 ONNX model.

Input ``images`` is (1, 3, H, W) with dynamic H/W (both multiples of 14). Specify
only the longest side via ``--res``; the engine accepts any HxW with both sides in
``[14, res]`` and is tuned for ``(res, res)``. The inference script reads this back
from the engine, so no input size is passed at run time.

For external-data models (giant / nested), keep the ``<name>.onnx_data`` sidecar in
the same folder as the ``.onnx``.

Example:
    python3 onnx2trt.py --onnx da3metric_large.onnx --saveEngine da3metric_large.engine \
        --fp16 --res 504
"""

import argparse

import tensorrt as trt

PATCH_SIZE = 14  # DINO patch size; every input side must be a multiple of this.


def build_engine(onnx_path, engine_path, res=504, fp16=False, workspace_gb=8):
    assert res % PATCH_SIZE == 0, f"res must be a multiple of {PATCH_SIZE}, got {res}"

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # parse_from_file (not parse(bytes)) so external-data sidecars are resolved.
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    # Longest side <= res, any aspect ratio / orientation; opt = (res, res).
    min_hw = (PATCH_SIZE, PATCH_SIZE)
    opt_hw = (res, res)
    max_hw = (res, res)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled.")
        else:
            print("FP16 requested but not supported on this platform; using FP32.")

    input_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (1, 3, min_hw[0], min_hw[1]),
        (1, 3, opt_hw[0], opt_hw[1]),
        (1, 3, max_hw[0], max_hw[1]),
    )
    config.add_optimization_profile(profile)
    print(f"Input profile: H/W in [{PATCH_SIZE}, {res}], opt ({res}x{res})")

    print(f"Building engine from {onnx_path} ... (this can take a while)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed.")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"Saved engine -> {engine_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="input ONNX path")
    parser.add_argument("--saveEngine", required=True, help="output .engine path")
    parser.add_argument("--fp16", action="store_true", help="enable FP16 precision")
    parser.add_argument("--res", type=int, default=504,
                        help="longest input side (multiple of 14); the engine accepts any "
                             "HxW with both sides in [14, res]")
    parser.add_argument("--workspace", type=int, default=8, help="workspace size in GiB")
    args = parser.parse_args()

    build_engine(
        args.onnx,
        args.saveEngine,
        res=args.res,
        fp16=args.fp16,
        workspace_gb=args.workspace,
    )
