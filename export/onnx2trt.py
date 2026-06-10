"""
Build a TensorRT engine (.engine) from a Depth-Anything-3 ONNX model.

The exported graph has a single input ``images`` of shape ``(1, 3, H, W)`` where H
and W are dynamic (the official pre-processing resizes the longest side to
``process_res`` while preserving aspect ratio, so the two spatial dims vary per
image). We therefore build an explicit-batch network with a single optimization
profile that covers a range of spatial sizes; pick ``--opt-*`` to match the size
you run most often (typically process_res x process_res).

Both H and W must stay multiples of 14 (the DINO patch size).

Example:
    python3 onnx2trt.py --onnx da3metric_large.onnx --saveEngine da3metric_large.engine \
        --fp16 --min 154 154 --opt 504 504 --max 504 504
"""

import argparse

import tensorrt as trt


def build_engine(onnx_path, engine_path, min_hw=(154, 154), opt_hw=(504, 504),
                 max_hw=(504, 504), fp16=False, workspace_gb=8):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    for hw in (min_hw, opt_hw, max_hw):
        assert hw[0] % 14 == 0 and hw[1] % 14 == 0, f"H/W must be multiples of 14, got {hw}"

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled.")
        else:
            print("FP16 requested but not supported on this platform; using FP32.")

    # The input tensor is assumed to be the first network input named "images".
    input_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (1, 3, min_hw[0], min_hw[1]),
        (1, 3, opt_hw[0], opt_hw[1]),
        (1, 3, max_hw[0], max_hw[1]),
    )
    config.add_optimization_profile(profile)

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
    parser.add_argument("--min", type=int, nargs=2, default=[154, 154], metavar=("H", "W"),
                        help="min input H W (multiples of 14)")
    parser.add_argument("--opt", type=int, nargs=2, default=[504, 504], metavar=("H", "W"),
                        help="optimal input H W (multiples of 14)")
    parser.add_argument("--max", type=int, nargs=2, default=[504, 504], metavar=("H", "W"),
                        help="max input H W (multiples of 14)")
    parser.add_argument("--workspace", type=int, default=8, help="workspace size in GiB")
    args = parser.parse_args()

    build_engine(
        args.onnx,
        args.saveEngine,
        min_hw=tuple(args.min),
        opt_hw=tuple(args.opt),
        max_hw=tuple(args.max),
        fp16=args.fp16,
        workspace_gb=args.workspace,
    )
