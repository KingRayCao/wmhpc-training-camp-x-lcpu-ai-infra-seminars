"""问题 5.1:per-tensor scale 与 outlier。

构造一个张量:一万个元素均匀分布在 [-1, 1],外加一个 3000 的
outlier。按 per-tensor 方式量化到 E4M3(scale = amax / 448,cast 用
torch.float8_e4m3fn),反量化后测逐点相对误差,填题面的表并回答三问。

需要动手的是下面两个 TODO;跑法:
    uv run python kernels/quant_outlier.py
输出直接用于报告,没有自动判测。
"""

import torch

E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0**-9


def build_tensor(n: int = 10000, outlier: float = 3000.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, generator=g) * 2 - 1
    return torch.cat([x, torch.tensor([outlier])])


def quant_dequant_per_tensor(x: torch.Tensor) -> torch.Tensor:
    """per-tensor E4M3 量化再反量化。

    步骤:算 scale = amax / 448;除 scale 后 cast 到
    torch.float8_e4m3fn;cast 回 float 再乘 scale。
    """
    scale_x = x.abs().max() / E4M3_MAX
    x_quant = (x / scale_x).to(torch.float8_e4m3fn)
    x_dequant = x_quant.to(torch.float32) * scale_x
    return x_dequant


def rel_err_at(x: torch.Tensor, y: torch.Tensor, value: float) -> float:
    """取 x 中最接近 value 的元素,返回该点的相对误差。

    表格的每一格都从这里计算。
    """
    idx = (x - value).abs().argmin()
    return (abs(x[idx] - y[idx]) / abs(x[idx])).item()


def main() -> None:
    x = build_tensor()
    y = quant_dequant_per_tensor(x)
    print("含 outlier:")
    for v in (0.5, 0.1, 0.01, 0.005, 3000.0):
        print(f"  x≈{v:<8} rel_err={rel_err_at(x, y, v):.3e}")
    # (a) 去掉 outlier 重新量化,对比 0.5 处的误差
    print("不含 outlier:")
    y_no_outlier = quant_dequant_per_tensor(x[:-1])
    for v in (0.5, 0.1, 0.01, 0.005):
        print(f"  x≈{v:<8} rel_err={rel_err_at(x[:-1], y_no_outlier, v):.3e}")
    # (b) 找出被量化成 0 的阈值,写出它与 scale 的关系式
    scale = x.abs().max() / E4M3_MAX
    threshold = 0.5 * E4M3_MIN_SUBNORMAL * scale
    print(f"被量化成 0 的阈值: {threshold:.3e}, scale = {scale:.3e}")
    # (c) 换 1x128 的 per-block scale,对比含/不含 outlier 的 block
    pad = (-x.numel()) % 128
    blocks = torch.cat([x, x.new_zeros(pad)]).reshape(-1, 128)
    scales = blocks.abs().amax(dim=1, keepdim=True) / E4M3_MAX
    y = (blocks / scales).to(torch.float8_e4m3fn).to(torch.float32) * scales
    print("per-block scale:")
    for v in (0.5, 0.1, 0.01, 0.005, 3000.0):
        flat_idx = (blocks.flatten()[: x.numel()] - v).abs().argmin()
        block_idx = flat_idx // 128
        error = rel_err_at(blocks[block_idx], y[block_idx], v)
        print(f"  x≈{v:<8} rel_err={error:.3e}")

    outlier_block = x.abs().argmax().item() // 128
    normal_block = 0 if outlier_block != 0 else 1
    print("block 对比:")
    compared_blocks = (("不含 outlier", normal_block), ("含 outlier", outlier_block))
    for label, block_idx in compared_blocks:
        valid = min(128, x.numel() - block_idx * 128)
        block_x = blocks[block_idx, :valid]
        block_y = y[block_idx, :valid]
        ordinary = torch.ones(valid, dtype=torch.bool)
        if block_idx == outlier_block:
            ordinary[block_x.abs().argmax()] = False
        rel = (
            (block_x[ordinary] - block_y[ordinary]).abs()
            / block_x[ordinary].abs()
        )
        zeroed = ((block_y[ordinary] == 0) & (block_x[ordinary] != 0)).sum()
        block_threshold = 0.5 * E4M3_MIN_SUBNORMAL * scales[block_idx, 0]
        print(
            f"  {label:<12} block={block_idx:2d} "
            f"amax={blocks[block_idx].abs().max():.3e} "
            f"scale={scales[block_idx, 0]:.3e} "
            f"zero_threshold={block_threshold:.3e} "
            f"zeroed={zeroed}/{ordinary.sum()} "
            f"ordinary_mean_rel_err={rel.mean():.3e}"
        )

if __name__ == "__main__":
    main()
