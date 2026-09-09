"""Generate the Module 6 TileLang lowering artifacts without running a GPU."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tilelang
from tilelang import tvm


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "assignment01"))

from kernels.tilelang_matmul import make_matmul  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "output"
TARGETS = ("sm_90a", "sm_100a")


def matching_lines(text: str, patterns: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for line in text.splitlines():
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
            stripped = line.strip()
            if stripped and stripped not in result:
                result.append(stripped)
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    func = make_matmul(
        256,
        256,
        128,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        threads=128,
        num_stages=3,
    )

    log_lines = [
        f"TileLang version: {tilelang.__version__}",
        "Kernel: assignment01/kernels/tilelang_matmul.py::make_matmul",
        "Shape: M=N=256, K=128",
        "Config: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, threads=128, num_stages=3",
    ]

    patterns = (
        r"wgmma",
        r"tcgen05",
        r"mma[._]sync",
        r"ptx_mma",
        r"ldmatrix",
        r"cp\.async",
        r"tma",
        r"utcmma",
        r"descriptor",
        r"swizzle",
    )

    for arch in TARGETS:
        target = tvm.target.Target({"kind": "cuda", "arch": arch})
        # TileLang 0.1.13's LowerTileOp pass also consults Target.current().
        with target:
            artifact = tilelang.lower(func, target=target, enable_device_compile=True)
        host_ir = artifact.host_mod.script()
        device_ir = artifact.device_mod.script()
        cuda_source = artifact.kernel_source

        (OUT_DIR / f"matmul_{arch}.host.tir").write_text(host_ir, encoding="utf-8")
        (OUT_DIR / f"matmul_{arch}.device.tir").write_text(device_ir, encoding="utf-8")
        (OUT_DIR / f"matmul_{arch}.cu").write_text(cuda_source, encoding="utf-8")

        log_lines.extend(
            [
                "",
                f"[{arch}]",
                f"normalized target: {artifact.target}",
                "device compilation: PASS",
                f"host IR lines: {len(host_ir.splitlines())}",
                f"device IR lines: {len(device_ir.splitlines())}",
                f"CUDA source lines: {len(cuda_source.splitlines())}",
                "key lowering lines:",
            ]
        )
        matches = matching_lines(host_ir + "\n" + device_ir + "\n" + cuda_source, patterns)
        log_lines.extend(f"  {line}" for line in matches)

    (OUT_DIR / "lowering.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
