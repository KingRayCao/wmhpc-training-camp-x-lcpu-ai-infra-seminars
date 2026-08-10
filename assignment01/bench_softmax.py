"""Benchmark Torch, Triton and TileLang softmax implementations."""

import torch
import triton

from kernels.softmax import softmax as triton_softmax
from kernels.tilelang_softmax import softmax as tilelang_softmax


def bench(M=4096, widths=(256, 1024, 4096)):
    assert torch.cuda.is_available(), "benchmark 需要 GPU"

    print(f"M={M}, dtype=float32（首次 JIT 编译不计时）")
    print(
        f"{'N':>6}  {'backend':<8}  {'latency/ms':>10}  "
        f"{'effective GB/s':>14}  {'vs torch':>9}"
    )

    for N in widths:
        x = torch.randn((M, N), device="cuda", dtype=torch.float32)

        # Compile/warm up both generated kernels before measuring them.
        tilelang_result = tilelang_softmax(x)
        triton_result = triton_softmax(x)
        torch_result = torch.softmax(x, dim=-1)
        torch.testing.assert_close(tilelang_result, torch_result, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(triton_result, torch_result, atol=1e-5, rtol=1e-5)

        timings = {
            "torch": triton.testing.do_bench(lambda: torch.softmax(x, dim=-1)),
            "triton": triton.testing.do_bench(lambda: triton_softmax(x)),
            "tilelang": triton.testing.do_bench(lambda: tilelang_softmax(x)),
        }
        torch_ms = timings["torch"]
        transferred_bytes = 2 * x.numel() * x.element_size()

        for backend, latency_ms in timings.items():
            effective_gbps = transferred_bytes / (latency_ms * 1e-3) / 1e9
            speedup = torch_ms / latency_ms
            print(
                f"{N:6d}  {backend:<8}  {latency_ms:10.4f}  "
                f"{effective_gbps:14.1f}  {speedup:8.2f}x"
            )


if __name__ == "__main__":
    bench()
