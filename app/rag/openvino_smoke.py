from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import openvino as ov
from openvino import opset13 as ops


def _build_relu_model() -> ov.Model:
    x = ops.parameter([1, 4], dtype=np.float32, name="x")
    y = ops.relu(x)
    return ov.Model([ops.result(y)], [x], "relu_smoke")


def _build_tiny_matmul_model() -> ov.Model:
    x = ops.parameter([1, 4], dtype=np.float32, name="x")
    weight_value = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5, 1.6],
        ],
        dtype=np.float32,
    )
    weight = ops.constant(weight_value)
    matmul = ops.matmul(x, weight, False, False)
    bias = ops.constant(np.array([0.01, -0.02, 0.03, -0.04], dtype=np.float32))
    biased = ops.add(matmul, bias)
    activated = ops.relu(biased)
    return ov.Model([ops.result(activated)], [x], "tiny_matmul_smoke")


def _run_case(case_name: str, *, device: str) -> dict[str, Any]:
    core = ov.Core()
    available_devices = list(core.available_devices)

    if case_name == "relu":
        model = _build_relu_model()
        input_data = np.array([[-1.0, 0.0, 2.5, -3.0]], dtype=np.float32)
    elif case_name == "tiny_matmul":
        model = _build_tiny_matmul_model()
        input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    else:
        raise ValueError(f"Unknown smoke case: {case_name}")

    compiled_model = core.compile_model(model, device)
    result = compiled_model([input_data])[0]
    return {
        "case": case_name,
        "device": device,
        "available_devices": available_devices,
        "model_name": model.get_name(),
        "input_shape": list(input_data.shape),
        "output_shape": list(result.shape),
        "output_sample": np.asarray(result).reshape(-1).tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal OpenVINO compile/infer smoke tests")
    parser.add_argument(
        "--device",
        default="GPU",
        help="OpenVINO target device, e.g. GPU or CPU",
    )
    parser.add_argument(
        "--case",
        choices=["relu", "tiny_matmul"],
        default="relu",
        help="Which minimal graph to compile and run",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = _run_case(args.case, device=args.device)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
