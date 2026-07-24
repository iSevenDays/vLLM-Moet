#!/usr/bin/env python3
"""Standalone static inspector and CUDA-driver checker for the SM89 W2 cubin."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

import numpy as np


KERNELS = {
    "k1024_n4096": ("moe_w2_sm89_decode_k1024_n4096", 1024, 4096),
    "k4096_n2048": ("moe_w2_sm89_decode_k4096_n2048", 4096, 2048),
}
W2_LEVELS = np.array([-4.0, -1.0, 1.0, 4.0], dtype=np.float32)
FP8_LEVELS = np.array(
    [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
    dtype=np.float32,
)
FP8_BITS = np.array(
    [0xC8, 0xC0, 0xB8, 0xB0, 0x00, 0x30, 0x38, 0x40, 0x48],
    dtype=np.uint8,
)


def inspect_cubin(cubin: pathlib.Path) -> None:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is not on PATH")

    resources = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(cubin)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    sass = subprocess.run(
        [cuobjdump, "--dump-sass", str(cubin)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    missing = [symbol for symbol, _, _ in KERNELS.values() if symbol not in sass]
    if missing:
        raise RuntimeError(f"missing cubin symbols: {', '.join(missing)}")
    fp8_mma = [
        line.strip()
        for line in sass.splitlines()
        if re.search(r"(?:Q|H)MMA\..*(?:E4M3|FP8)", line)
    ]
    if not fp8_mma:
        raise RuntimeError("no E4M3/FP8 tensor-core instruction found")

    print(resources.rstrip())
    print("\nFP8 tensor-core sample:")
    for line in fp8_mma[:12]:
        print(line)
    print(
        f"\nStatic gate passed: {len(fp8_mma)} FP8 tensor-core "
        "instructions found"
    )


class CudaError(RuntimeError):
    pass


class CudaDriver:
    def __init__(self, device_ordinal: int) -> None:
        library = ctypes.util.find_library("cuda") or "libcuda.so.1"
        self.lib = ctypes.CDLL(library)
        self._bind()
        self._check(self.lib.cuInit(0), "cuInit")
        device = ctypes.c_int()
        self._check(
            self.lib.cuDeviceGet(ctypes.byref(device), device_ordinal),
            "cuDeviceGet",
        )
        self.context = ctypes.c_void_p()
        self._check(
            self.lib.cuCtxCreate_v2(ctypes.byref(self.context), 0, device),
            "cuCtxCreate_v2",
        )

    def _bind(self) -> None:
        lib = self.lib
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuCtxCreate_v2.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        lib.cuModuleLoad.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
        ]
        lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.cuMemAlloc_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        lib.cuMemcpyHtoD_v2.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.cuMemcpyDtoH_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
        ]
        lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.cuCtxSynchronize.argtypes = []
        lib.cuEventCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        lib.cuEventDestroy_v2.argtypes = [ctypes.c_void_p]
        lib.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        lib.cuEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        lib.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        name = ctypes.c_char_p()
        message = ctypes.c_char_p()
        self.lib.cuGetErrorName(result, ctypes.byref(name))
        self.lib.cuGetErrorString(result, ctypes.byref(message))
        error_name = name.value.decode() if name.value else f"CUDA_ERROR_{result}"
        error_message = message.value.decode() if message.value else "unknown error"
        raise CudaError(f"{operation}: {error_name}: {error_message}")

    def load_module(self, cubin: pathlib.Path) -> ctypes.c_void_p:
        module = ctypes.c_void_p()
        self._check(
            self.lib.cuModuleLoad(
                ctypes.byref(module), str(cubin).encode("utf-8")
            ),
            "cuModuleLoad",
        )
        return module

    def get_function(
        self, module: ctypes.c_void_p, symbol: str
    ) -> ctypes.c_void_p:
        function = ctypes.c_void_p()
        self._check(
            self.lib.cuModuleGetFunction(
                ctypes.byref(function), module, symbol.encode("ascii")
            ),
            f"cuModuleGetFunction({symbol})",
        )
        return function

    def upload(self, array: np.ndarray) -> int:
        contiguous = np.ascontiguousarray(array)
        pointer = ctypes.c_uint64()
        self._check(
            self.lib.cuMemAlloc_v2(ctypes.byref(pointer), contiguous.nbytes),
            "cuMemAlloc_v2",
        )
        self._check(
            self.lib.cuMemcpyHtoD_v2(
                pointer.value,
                ctypes.c_void_p(contiguous.ctypes.data),
                contiguous.nbytes,
            ),
            "cuMemcpyHtoD_v2",
        )
        return pointer.value

    def allocate(self, size: int) -> int:
        pointer = ctypes.c_uint64()
        self._check(
            self.lib.cuMemAlloc_v2(ctypes.byref(pointer), size),
            "cuMemAlloc_v2",
        )
        return pointer.value

    def download(self, pointer: int, shape: tuple[int, ...]) -> np.ndarray:
        output = np.empty(shape, dtype=np.uint16)
        self._check(
            self.lib.cuMemcpyDtoH_v2(
                ctypes.c_void_p(output.ctypes.data), pointer, output.nbytes
            ),
            "cuMemcpyDtoH_v2",
        )
        return output

    def launch(
        self,
        function: ctypes.c_void_p,
        descriptor_pointer: int,
        grid_x: int,
        grid_y: int,
    ) -> None:
        argument = ctypes.c_uint64(descriptor_pointer)
        arguments = (ctypes.c_void_p * 1)(
            ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
        )
        self._check(
            self.lib.cuLaunchKernel(
                function,
                grid_x,
                grid_y,
                1,
                64,
                1,
                1,
                0,
                ctypes.c_void_p(),
                arguments,
                None,
            ),
            "cuLaunchKernel",
        )

    def synchronize(self) -> None:
        self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def elapsed_ms(self, launches, runs: int) -> float:
        start = ctypes.c_void_p()
        stop = ctypes.c_void_p()
        self._check(self.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate")
        self._check(self.lib.cuEventCreate(ctypes.byref(stop), 0), "cuEventCreate")
        try:
            self._check(
                self.lib.cuEventRecord(start, ctypes.c_void_p()),
                "cuEventRecord(start)",
            )
            for _ in range(runs):
                launches()
            self._check(
                self.lib.cuEventRecord(stop, ctypes.c_void_p()),
                "cuEventRecord(stop)",
            )
            self._check(
                self.lib.cuEventSynchronize(stop), "cuEventSynchronize"
            )
            elapsed = ctypes.c_float()
            self._check(
                self.lib.cuEventElapsedTime(
                    ctypes.byref(elapsed), start, stop
                ),
                "cuEventElapsedTime",
            )
            return elapsed.value / runs
        finally:
            self.lib.cuEventDestroy_v2(start)
            self.lib.cuEventDestroy_v2(stop)

    def free(self, pointer: int) -> None:
        self._check(self.lib.cuMemFree_v2(pointer), "cuMemFree_v2")

    def unload(self, module: ctypes.c_void_p) -> None:
        self._check(self.lib.cuModuleUnload(module), "cuModuleUnload")

    def close(self) -> None:
        if getattr(self, "context", None):
            self._check(
                self.lib.cuCtxDestroy_v2(self.context), "cuCtxDestroy_v2"
            )
            self.context = ctypes.c_void_p()


@dataclass
class HostCase:
    activation_bits: np.ndarray
    activation_scales: np.ndarray
    packed_weights: np.ndarray
    packed_weight_scales: np.ndarray
    reference: np.ndarray


def pack_weights(codes: np.ndarray) -> np.ndarray:
    n, k = codes.shape
    lanes = (
        codes.reshape(n // 16, 2, 8, k // 64, 2, 2, 4, 4)
        .transpose(0, 3, 2, 6, 1, 4, 5, 7)
        .reshape(-1, 4)
        .astype(np.uint8)
    )
    return (
        lanes[:, 0]
        | (lanes[:, 1] << 2)
        | (lanes[:, 2] << 4)
        | (lanes[:, 3] << 6)
    ).astype(np.uint8)


def pack_weight_scales(scales: np.ndarray) -> np.ndarray:
    n, k_groups = scales.shape
    return (
        scales.reshape(n // 16, 16, k_groups)
        .transpose(0, 2, 1)
        .reshape(-1)
        .astype(np.uint8)
    )


def make_case(seed: int, m: int, k: int, n: int) -> HostCase:
    rng = np.random.default_rng(seed)
    activation_indices = rng.integers(
        0, len(FP8_LEVELS), size=(m, k), dtype=np.uint8
    )
    activation_bits = FP8_BITS[activation_indices]
    activation_values = FP8_LEVELS[activation_indices]
    activation_exponents = rng.integers(-6, 2, size=(m, k // 32))
    activation_scales = np.ldexp(
        np.ones((m, k // 32), dtype=np.float32), activation_exponents
    )

    weight_codes = rng.integers(0, 4, size=(n, k), dtype=np.uint8)
    weight_exponents = rng.integers(-8, 1, size=(n, k // 32), dtype=np.int16)
    weight_scale_bytes = (weight_exponents + 127).astype(np.uint8)
    weight_scales = np.ldexp(
        np.ones((n, k // 32), dtype=np.float32), weight_exponents
    )

    activation_dequantized = (
        activation_values.reshape(m, k // 32, 32)
        * activation_scales[:, :, None]
    ).reshape(m, k)
    weight_dequantized = (
        W2_LEVELS[weight_codes].reshape(n, k // 32, 32)
        * weight_scales[:, :, None]
    ).reshape(n, k)
    reference = activation_dequantized @ weight_dequantized.T

    return HostCase(
        np.ascontiguousarray(activation_bits),
        np.ascontiguousarray(activation_scales.astype(np.float32)),
        np.ascontiguousarray(pack_weights(weight_codes)),
        np.ascontiguousarray(pack_weight_scales(weight_scale_bytes)),
        np.ascontiguousarray(reference.astype(np.float32)),
    )


def bfloat16_to_float32(values: np.ndarray) -> np.ndarray:
    words = values.astype(np.uint32) << 16
    return words.view(np.float32)


def run_case(
    driver: CudaDriver,
    module: ctypes.c_void_p,
    kernel_key: str,
    m: int,
    pairs: int,
    warmup: int,
    runs: int,
    seed: int,
) -> None:
    symbol, k, n = KERNELS[kernel_key]
    function = driver.get_function(module, symbol)
    allocations: list[int] = []
    descriptors = np.zeros((pairs, 6), dtype=np.uint64)
    references: list[np.ndarray] = []
    outputs: list[int] = []

    try:
        for pair in range(pairs):
            case = make_case(seed + pair, m, k, n)
            activation = driver.upload(case.activation_bits)
            activation_scale = driver.upload(case.activation_scales)
            weight = driver.upload(case.packed_weights)
            weight_scale = driver.upload(case.packed_weight_scales)
            output = driver.allocate(m * n * 2)
            allocations.extend(
                [activation, activation_scale, weight, weight_scale, output]
            )
            outputs.append(output)
            references.append(case.reference)
            descriptors[pair] = [
                activation,
                activation_scale,
                weight,
                weight_scale,
                output,
                m,
            ]

        descriptor_pointer = driver.upload(descriptors)
        allocations.append(descriptor_pointer)

        def launch() -> None:
            driver.launch(function, descriptor_pointer, n // 64, pairs)

        for _ in range(warmup):
            launch()
        driver.synchronize()

        launch()
        driver.synchronize()
        worst_relative = 0.0
        worst_rmse = 0.0
        for pair, output in enumerate(outputs):
            actual = bfloat16_to_float32(driver.download(output, (m, n)))
            reference = references[pair]
            error = actual - reference
            relative = float(
                np.max(np.abs(error)) / max(1.0, np.max(np.abs(reference)))
            )
            rmse = float(
                np.sqrt(np.mean(np.square(error)))
                / max(1.0, np.sqrt(np.mean(np.square(reference))))
            )
            worst_relative = max(worst_relative, relative)
            worst_rmse = max(worst_rmse, rmse)

        if worst_relative > 0.02 or worst_rmse > 0.01:
            raise AssertionError(
                f"{kernel_key}: correctness failed: "
                f"max-relative={worst_relative:.6f}, rmse={worst_rmse:.6f}"
            )

        elapsed_ms = driver.elapsed_ms(launch, runs)
        tflops = 2.0 * pairs * m * n * k / (elapsed_ms * 1.0e9)
        packed_gbps = pairs * n * k / 4.0 / (elapsed_ms * 1.0e6)
        print(
            f"{kernel_key}: correctness PASS "
            f"(max-relative={worst_relative:.6f}, rmse={worst_rmse:.6f})"
        )
        print(
            f"{kernel_key}: {elapsed_ms:.4f} ms, "
            f"{tflops:.3f} TFLOP/s, {packed_gbps:.2f} packed-weight GB/s"
        )
    finally:
        for pointer in reversed(allocations):
            driver.free(pointer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the SM89 W2 cubin without a GPU, or explicitly run its "
            "correctness/performance gates through the CUDA driver API."
        )
    )
    parser.add_argument("--cubin", type=pathlib.Path, required=True)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Run cuobjdump static gates only; do not initialize CUDA.",
    )
    parser.add_argument(
        "--kernel",
        choices=["both", *KERNELS],
        default="both",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.cubin.is_file():
        raise FileNotFoundError(args.cubin)
    inspect_cubin(args.cubin)
    if args.inspect_only:
        print("CUDA was not initialized; no GPU work was submitted.")
        return 0
    if args.pairs < 1 or args.runs < 1 or args.warmup < 0:
        raise ValueError("pairs/runs must be positive and warmup non-negative")

    selected = list(KERNELS) if args.kernel == "both" else [args.kernel]
    driver = CudaDriver(args.device)
    module = driver.load_module(args.cubin)
    try:
        for index, kernel_key in enumerate(selected):
            run_case(
                driver,
                module,
                kernel_key,
                args.m,
                args.pairs,
                args.warmup,
                args.runs,
                args.seed + index * 1000,
            )
    finally:
        driver.unload(module)
        driver.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CudaError, RuntimeError, AssertionError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
