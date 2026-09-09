# 作业 2 Tensor Core & Pipeline 答案

## Module 0 环境与峰值

### 0. 口径约定（先声明，再计算）

开始推导前，按题面要求明确统一口径。

| 项目 | 本报告采用 | 备注 |
| --- | --- | --- |
| dense / sparse | **dense（非稀疏）** | 2:4 结构化稀疏在 dense 基础上 ×2；NVFP4 例外（见下） |
| boost / base 频率 | **boost** | 峰值按 boost 频率计算；B300 存在 boost 口径差（见 0.2） |
| FMA 计法 | **1 FMA = 2 FLOP** | 分子恒为 `2·M·N·K` |
| 峰值单位 | TFLOPS（dense） | `FLOP/cycle/SM × SM × clock` |

---

### prob 0.1（HANDS-ON）`cuda/m0_env/01_first_mma.cu`

**目标**

编译并运行一个“单 warp 执行一条 `mma.sync.m16n8k16`”的最小 Tensor Core 程序；
分别在 5090（`ARCH=120a`）与 B300（默认 `sm_100f`）上跑通，再**故意用不匹配的 ARCH 编译运行一次**，
记录现象并结合 assignment01 Module 8 的 fatbin/JIT 内容解释。

#### 0.1.1 程序做什么

`m16n8k16` = 输出 16×8 的 `C`，A 为 16×16、B 为 16×8，K=16。
每个 warp 的 32 个 lane 各持 A/B/C fragment 的一部分（寄存器），全 warp 一致执行一条 `mma.sync`，
硬件完成 `C += A×B` 的乘加，由此验证 fragment 装载、mma.sync 和结果写回组成的最简 Tensor Core 通路。

#### 0.1.2 预期运行方式

```bash
cd assignment02/cuda
make run/m0_env/01_first_mma            # B300：默认 ARCH=100f
ARCH=120a make run/m0_env/01_first_mma  # 5090
make ptx/m0_env/01_first_mma            # 查看生成的 PTX（含 mma.sync.aligned.m16n8k16）
```

#### 0.1.3 不匹配 ARCH 的实测现象与 fatbin 解释

使用默认 `ARCH=100f` 在 B300 上运行，结果为

```text
D[0][0]=2 D[0][7]=2 D[15][0]=2 D[15][7]=2
```

Makefile 使用 `-gencode arch=compute_XX,code=sm_XX`。这里的
`arch=compute_XX` 指定 PTX 前端采用的虚拟指令集，`code=sm_XX` 指定最终写入
fatbin 的目标 cubin。当前命令没有 `code=compute_XX`，因此可执行文件不包含用于
运行时 JIT 的 PTX fallback。

在 B300 上运行为 RTX 5090 编译的 `sm_120a` 产物，得到以下实测错误。

```text
CUDA error cudaErrorNoKernelImageForDevice at m0_env/01_first_mma.cu:76:
no kernel image is available for execution on the device
```

驱动找不到与 B300 匹配的 cubin，fatbin 中又没有可供 JIT 的 PTX，于是 kernel
无法加载。若希望保留向新架构 JIT 的可能，需要额外加入
`-gencode arch=compute_XX,code=compute_XX`。独立执行
`make ptx/m0_env/01_first_mma` 只会生成便于检查的 PTX 文件，不会改变已经链接的
可执行文件。

---

### prob 0.2（DERIVE）Tensor Core 理论峰值推导

#### 0.2.1 推导链（沿用课上 A100 方法，S018-S019）

课上 A100 的推导（S018-S019）是一条乘法链。

```
一条 mma 的 FLOP（m16n8k16 → 2048 MAC × 2 = 4096 FLOP）
  ÷ mma 延迟 8 cycle
  = 512 FLOP/cycle/子分区        （每子分区 = 1 组 Tensor Core）
  × 4 子分区
  = 2048 FLOP/cycle/SM
  × 108 SM
  = 221,184 FLOP/cycle
  × boost 1.41 GHz
  = 311.9 TFLOPS   （A100 FP16 dense，与 datasheet 312 一致）
```

把 `FLOP/cycle/SM` 记为**每 SM 每周期有效吞吐**，对任意 GPU 都有

```
bf16 峰值(dense) = bf16 FLOP/cycle/SM  ×  SM 数  ×  boost 频率
fp8 峰值 ≈ 2 × bf16 峰值        （dtype 位宽减半 → 吞吐 ×2）
fp4 峰值 ≈ 4 × bf16 峰值        （dtype 位宽减半 → 吞吐 ×4）
机器平衡点 = bf16 峰值 / 显存带宽   （单位 FLOP/byte，同 S020–S021）
```

其中 `bf16 FLOP/cycle/SM = 峰值 ÷ (SM × clock)`，等价于“每 SM 张量核数 × 每核每周期 FMA 数 × 2”。

课件给出的代际峰值可作为锚点，分别是 **A100 312 / H100 990 / B200 2250（FP16 dense TFLOPS）**。
本作业要推导的 5090、B300 分别是消费级 Blackwell 与 Blackwell Ultra，均不是课件列出的 B200，需按上述链独立推导。

#### 0.2.2 关键硬件参数（Firecrawl 抓取）

| 参数 | RTX 5090（GB202, sm_120a） | B300（Blackwell Ultra, sm_100f） |
| --- | --- | --- |
| SM 数 | **170** | **160** |
| Tensor Core（5 代） | 680（4/SM） | 640（4/SM） |
| boost 频率 | **2407 MHz**（NVIDIA 官方 2.41 GHz；base 2.01 GHz） | **~2032 MHz**（TechPowerUp B300 SXM6；NVIDIA 峰值口径约 1.9 GHz） |
| 显存 | 32 GB GDDR7，512-bit | 288 GB HBM3e，8192-bit |
| 带宽 | **1792 GB/s** | **8 TB/s（8000 GB/s）** |
| 工艺/晶体管 | TSMC 4NP / 92.2B | TSMC 4NP / 208B（双 die） |

#### 0.2.3 推导过程

**RTX 5090**
- 4 TC/SM，每 TC 每周期 128 FMA（bf16，消费级 Blackwell 5 代）→ `4 × 128 × 2 = 1024 FLOP/cycle/SM`。
- `1024 × 170 × 2407 MHz = 419.0 TFLOPS`（dense bf16）。
- fp8 = 2× → **838 TFLOPS**；fp4 = 4× → **1676 TFLOPS**（dense）。

**B300（Blackwell Ultra）**
- 峰值(dense bf16) = **2500 TFLOPS = 2.5 PFLOPS**（NVIDIA 官方 fp8 dense 5 PF ÷ 2；GB300 NVL72 规格表 bf16 亦为 2500）。
- 反推每 SM 有效吞吐得到 `2500e12 ÷ (160 SM × 1.907 GHz) = 8192 FLOP/cycle/SM`（dense bf16；即 4 TC/SM × 1024 FMA × 2）。
  若用 TechPowerUp 的 2032 MHz boost 硬乘，则等效约 7689 FLOP/cycle/SM、峰值约 2.66 PF，二者差 ~6%，属频率口径（见 0.2.5）。
- 相对课件里的 B200（2250 dense），B300 以相近架构配合更高时钟，按 `2250 × 2032/1830` 估算约为 2500 TFLOPS。
- fp8 = 2× → **5000 TFLOPS = 5 PFLOPS**；fp4（按位宽 4× 估算）→ **10 PFLOPS**。

#### 0.2.4 结果表

| 量 | 5090 | B300 |
| --- | --- | --- |
| bf16 FLOP/cycle/SM（dense） | 1024 | 8192 |
| bf16 峰值（TFLOPS，dense） | **419** | **2500（2.5 PF）** |
| fp8 峰值（TFLOPS，dense） | **838** | **5000（5 PF）** |
| fp4 峰值（TFLOPS，dense） | **1676** | 10 000（位宽 4× 估算）/ **15 000（datasheet）** |
| datasheet 对照值与口径差异 | NVIDIA “AI TOPS **3352**” = fp4 **sparse**（= 2×1676）；我算的是 dense | NVIDIA fp4 dense **15 PF**、fp8 dense 5 PF、NVFP4 sparse/dense=**4/3** |
| HBM/GDDR 带宽（GB/s） | 1792 | 8000（8 TB/s） |
| 机器平衡点（FLOP/byte，bf16） | **234** | **312.5** |

#### 0.2.5 datasheet 对照与口径差异说明

1. **dense vs sparse**。NVIDIA 消费级页只标 “AI TOPS”（= fp4 **sparse**，2-out-of-4 结构稀疏 ×2）。
   RTX 5090 的 3352 TOPS 反推 dense，计算过程为 `3352/2 = 1676(fp4) → 838(fp8) → 419(bf16)`，与上面的推导一致。
2. **boost vs base 频率（B300）**。按 TechPowerUp 的 2032 MHz 与 8192 FLOP/cycle/SM 硬乘得 2.66 PF，
   NVIDIA 官方峰值 2.5 PF 对应的有效频率约 1.9 GHz。约 6% 的差异来自**峰值口径所用频率**（boost 上限 vs 规格化典型频率）。
3. **NVFP4 特殊口径（B300 fp4）**。按位宽 4× 估算 fp4 = 10 PF，但 datasheet 给 15 PF。
   Blackwell Ultra 对 **NVFP4 有 1.5× 专属加速**（第五代 TC + 第二代 Transformer Engine），
   其 **sparse/dense = 4/3**（15 PF × 4/3 = 20 PF sparse），不同于普通 2-out-of-4 结构稀疏的 2 倍关系。
   故 fp4 一行“估算 10 PF / datasheet 15 PF”的差距正是这一口径。

#### 0.2.6 机器平衡点与单条 MMA

机器平衡点使用 dense bf16 峰值除以显存带宽。单位换算中
`1 TFLOPS / 1 GB/s = 1000 FLOP/byte`，因此

$$
I_{5090}=\frac{419\times1000}{1792}\approx234\ \mathrm{FLOP/byte},
$$

$$
I_{B300}=\frac{2500\times1000}{8000}=312.5\ \mathrm{FLOP/byte}.
$$

单条 `m16n8k16` fp16 MMA 完成

$$
2MNK=2\times16\times8\times16=4096\ \mathrm{FLOP}.
$$

按题面口径，它读取 512 B 的 A、256 B 的 B，并写回 512 B 的 fp32 D，
总流量为 1280 B，计算强度为

$$
AI_{\rm mma}=\frac{4096}{1280}=3.2\ \mathrm{FLOP/byte}.
$$

3.2 FLOP/byte 只达到 B300 机器平衡点的约 1%。如果每条 MMA 都直接从
HBM 读取 A/B 并写回 D，带宽最多支撑

$$
3.2\times8000=25.6\ \mathrm{TFLOPS},
$$

Tensor Core 会长期等待数据。后续 M2 到 M4 使用 shared-memory staging、
swizzle、TMA 和多级流水，让一个 tile 服务更多计算，并把搬运与计算重叠。
这些机制共同提高 kernel 层面的数据复用和供数效率。

---

### prob 0.3（CONCEPT）判断正误 + 一句理由

**(a) 正确。** 一条 mma 的计算强度以 `2MNK` 为分子，分母按 A、B 读入与 D 写回的字节总和计算。FMA 计作 2 FLOP，这正是 S016 的单条 mma 算术强度口径。课件中的 m16n8k16 计算为 `4096 FLOP ÷ (读A 512B + 读B 256B + 写D 512B = 1280 B) = 3.2 FLOP/byte`。

**(b) 正确。** mma.sync 是 warp 级协作指令，32 lane 各持 fragment 一部分，要求全 warp 一致执行。有 lane 发散时行为未定义。
A/B/C/D fragment 分布在 32 个 lane 的寄存器里，`mma.sync` 必须全 warp 收敛执行，发散即 UB。

**(c) 错误。** 增大 mma 的 M/N/K 会提高单条指令计算强度，同时会扩大 fragment，增加每 lane 的寄存器占用并降低 occupancy。可用形状也受指令集、寄存器和 TMEM 容量约束。

**(d) 错误。** 机器平衡点（roofline 拐点）针对 **kernel 级算术强度**，单条 mma 指令级强度不能直接决定整个 GEMM 的上限。
单条 m16n8k16 的强度仅约 3.2 FLOP/byte（远低于 5090 的 234），但 GEMM 通过 tiling 复用 A/B tile，
把有效访存摊薄，使 kernel 级强度越过平衡点后仍能逼近峰值。二者不可混用。

---

### 资料来源

- [GeForce RTX 5090 Graphics Cards，NVIDIA 官方](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)（boost 2.41 GHz、21760 CUDA、1792 GB/s、AI TOPS 3352）
- [NVIDIA RTX 5090 Specs，Spheron](https://www.spheron.network/blog/nvidia-rtx-5090-specs/)（170 SM、680 TC、FP16/FP8/FP4 sparse 表）
- [Inside NVIDIA Blackwell Ultra，NVIDIA 开发者博客](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)（160 SM、640 TC、FP8 dense 5 PF、NVFP4 dense 15 PF、8 TB/s）
- [NVIDIA B300 (Blackwell Ultra)，Spheron](https://www.spheron.network/blog/nvidia-b300-blackwell-ultra-guide/)（FP16/FP8/FP4 dense 对照表）
- [NVIDIA B300，Glenn K. Lockwood](https://www.glennklockwood.com/garden/processors/b300)（GB300 bf16 2500 / fp8 5000 / fp4 15000 dense）
- [NVIDIA B300 SXM6 AC，TechPowerUp](https://www.techpowerup.com/gpu-specs/b300-sxm6-ac.c4375)（boost 2032 MHz、NVFP4 sparse/dense=4/3）
- [NVIDIA DGX B300，NVIDIA 官方](https://www.nvidia.com/en-us/data-center/dgx-b300/)（FP4/FP8 系统级 dense/sparse 口径）

## Module 1：sm80 fragment 与 mma.sync

### 1.1 fragment 映射

由于一个寄存器内元素沿 K 连续，`ldmatrix` 的行加载/转置变体必须让 shared-memory 的 16B 对齐行布局与该 fragment 的 K 向打包对应；否则需要额外的寄存器重排。

### 1.2 DEBUG：错误 fragment

运行未修改版本的现象（待填写）：

```text
命令：make run/m1_sm80/02_bug_fragment
输出：
```

修复说明：A fragment 中 `a2/a3` 和 `a6/a7` 被错误地重复装载了 `a0/a1` 和 `a4/a5`。这两对值应来自 `group + 8` 行，即在线性地址上增加 `8 * 16`；当前错误使 D 的 8--15 行重复使用 A 的 0--7 行，因此 D 的上半 8 行正确、下半 8 行会等于对应的上半输出而不是参考值。具体不相等位置和数量以第一次运行输出填写。

| 项目 | 填写 |
| --- | --- |
| 错误 D 位置 | |
| 错误值与正确值的关系 | |
| 修改后的判测输出 | |

### 1.3 手写单 tile fp8 mma（FROM-SCRATCH）

实现要点：使用 1.1 的四个映射公式手工装载 A/B，调用 `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`，以 seed 生成可由 e4m3 精确表示的小整数，并与 CPU 参考严格比较。

| seed | 判测输出 |
| ---: | --- |
| | |
| | |
| | |
| | |
| | |

### 1.4 `ldmatrix`

| 路径 | 判测输出 |
| --- | --- |
| 手工装载 | |
| `ldmatrix` 装载 | |

观察记录：

* `ldmatrix` 将各 lane 的 shared-memory 地址生成、按 lane 分发和矩阵布局重排交给专用指令，减少显式标量/向量 load、地址计算和寄存器拼接。
* 手工路径必须显式计算每个 lane 的 fragment 坐标、执行普通 shared load，并把值打包到 mma 所需寄存器；这些工作无法由编译器凭空省去。

| 生成 PTX/SASS 的命令 | smem→fragment load 指令数 | 地址计算指令数 |
| --- | ---: | ---: |
| 手工路径 | | |
| `ldmatrix` 路径 | | |

### 1.5 `ldmatrix` 行跨度实验

预测依据：bank 宽度为 4 B；每个 16B chunk 覆盖 4 个 bank。行基址跨步改变后，同一列 chunk 在一个周期内映射到的 bank 可能重复。将预测比例写成相对 32B 档位的比例，实测栏保持空白。

| 档位 | 预测 wavefront 比 | 实测 wavefront | 实测 conflict | 平均 cycle |
| --- | ---: | ---: | ---: | ---: |
| 32 B | 1× | | | |
| 64 B | 2× | | | |
| 128 B | 4× | | | |
| 128+16 B | 1× | | | |

回答：

* wavefront 增加到 4 倍的是 **128 B** 行跨度；额外 padding 16 B 会打散重复的 bank 映射。
* wavefront 数是每条 shared load 的内部服务批次，不等于整个 kernel 的串行时间。8 个 warp 可以交错发射并隐藏部分批次/访存延迟，所以总耗时增幅通常小于 4 倍；还会受到指令调度、其他 warp 工作和测量噪声影响。

## Module 2：descriptor 与 swizzle

### 2.1 概念与排序

正确顺序：

`st.shared` → `fence.proxy.async` → `wgmma.fence` → `wgmma.mma_async` → `wgmma.commit_group` → `wgmma.wait_group`。

普通 `st.shared` 通过 generic proxy；`fence.proxy.async` 使写入对 async proxy 可见；`wgmma.fence` 建立 warpgroup 发射顺序；`commit_group` 把此前发射的异步 wgmma 划入一个完成组；`wait_group` 才等待该组完成后继续。

1. **错。** proxy fence 解决 generic proxy 与 async proxy 的可见性；TMA、tcgen05 等异步消费者同样可能需要它。
2. **错。** `commit_group` 只提交/封组，不等待全部 wgmma 完成；等待由 `wait_group` 完成。
3. **对。** 不做 proxy fence 时，async proxy 可能观察不到 generic proxy 的最新 shared-store 值。

### 2.2 SM100 descriptor

编码公式：

```text
d[13:0]   = (start_address >> 4) & 0x3fff
d[29:16]  = (LBO           >> 4) & 0x3fff
d[45:32]  = (SBO           >> 4) & 0x3fff
d[47:46]  = 1                 # version
d[63:61]  = layout            # NONE=0, 128B=2, 64B=4, 32B=6
```

对 64×64 bf16、8 行×16B atom：无 swizzle 时 LBO=128 B、SBO=1024 B；swizzle 时 LBO 被忽略取 0，SBO=1024 B。

| 场景 | LBO (B) | SBO (B) | layout | `make_desc` 结果 |
| --- | ---: | ---: | ---: | --- |
| K-major，无 swizzle，addr=0x1000 | 128 | 1024 | 0 | |
| K-major，128B swizzle，addr=0x2000 | 0 | 1024 | 2 | |
| MN-major，128B swizzle，addr=0x3000 | 0 | 1024 | 2 | |

场景 2/3 的 descriptor 相同，是因为 swizzle 模式下 descriptor 只描述同一 canonical 物理布局，且 LBO 被忽略；它本身不编码矩阵的 major 语义。MN-major/K-major 的区别由 MMA 的 operand/idesc（以及 staging 时对应的逻辑坐标解释）告诉 Tensor Core，而不是体现在这 64 位 descriptor 中。

### 2.3 swizzle 映射

通用形式（`rowBytes` 为 128/64/32，周期行数 `period` 为 8/4/2）：

```text
atom = row / period
r    = row % period
chunk = colByte / 16
in16  = colByte % 16
offset = atom * (period * rowBytes)
       + r * rowBytes
       + ((chunk ^ r) * 16)
       + in16
```

因此：128B 使用 period=8，64B 使用 period=4，32B 使用 period=2。判测输出：

| 模式 | 输出 |
| --- | --- |
| 128B | |
| 64B | |
| 32B | |

## Module 3：SM100 tcgen05

### 3.1 概念判断

* **(a) 对（按题面给定的 `tcgen05.ld` 映射）。** 每个 warp 的 lane 读取其对应的 TMEM 行片段，不能用另一个 warp 的 lane 直接替代。
* **(b) 对。** `tcgen05.mma` 由单线程发射，运算在硬件中异步执行并通过 mbarrier 通知完成。
* **(c) 错。** TMEM 结果需先用 `tcgen05.ld` 读入寄存器，再由线程写回 global memory；TMA 不能直接把 TMEM 当作 global source。
* **(d) 对。** 总容量为 \(128\times512\times4\) B；m128n256 的 f32 accumulator 为 \(128\times256\times4\) B，正好一半。
* **(e) 错。** `tcgen05.commit` 是非阻塞提交；必须等待关联 mbarrier 完成，并在适当 fence 后才能安全 `ld`。

### 3.2 单 tile GEMM

数据路径：`global → shared (K-major + 128B swizzle) → tcgen05.mma → TMEM → tcgen05.ld → global`。

| 项目 | 结果 |
| --- | --- |
| `make run/m3_tcgen05/02_single_tile` | |
| `./judge_tile.sh` | |
| 去掉 `fence.proxy.async` 后的现象 | |

去掉 proxy fence 会使 generic proxy 的 shared stores 与 tcgen05 async proxy 的读取之间缺少可见性排序，因而可能读到旧数据，表现为随机/成块的 MISMATCH；具体 seed、错误数量和是否复现填写实测结果。

### 3.3 mbarrier DEBUG

先运行错误版本并记录：

| rounds | 输出/是否超时 |
| ---: | --- |
| 1 | PASS |
| 2 | 超时 |
| 4 | 超时 |

正确状态机：初始化 phase=0、arrival count=1；每轮 `commit.arrive` 使计数满足后放行，等待该轮 phase；下一轮 phase 翻转（0→1→0…），并重新使用新的 arrival 计数。错误版本若始终等待 phase=0，第二轮起可能过早放行或永久等待；同时 `tcgen05.ld` 本身必须等待尚未完成的 mma，不能只依赖错误的 barrier phase。

### 3.4 CTA pair

(a) `cta_group::1` 的每个 CTA 保存完整 B tile；`cta_group::2` 将 N 对半切分，因此每个 CTA 的 B shared memory 约为前者的 **1/2**。两种实现每 CTA 都分配 64 个 TMEM columns，因而每 CTA 的 TMEM 占用不变；两个 CTA 合计的任务总 TMEM 容量也不变。

实际打印：`smem/block: cta_group::1 = 24588 B, cta_group::2 = 20492 B`，正好相差一半的 B 矩阵大小。

(b) 使用以下命令进行统计：

```sh
ncu \
  --kernel-name-base demangled \
  --kernel-name "regex:tile_kernel.*1" \
  --launch-count 1 \
  --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum \
  ./bin/m3_tcgen05/04_cta_pair
```

Nsight Compute 流量：

| 变体 | shared load wavefront | shared store wavefront | 合计 |
| --- | ---: | ---: | ---: |
| `cta_group::1` | 26 | 778 | 804 |
| `cta_group::2` | 29 | 650 | 679 |

(c) 省出的 shared memory 可用于增加 pipeline stage、扩大 tile、放置双缓冲/更多 TMA 暂存，从而提高预取与计算重叠。

(d) 依赖 SM90 引入的 **thread-block cluster / 2-CTA cooperative MMA（cluster multicast/DSM 协作）**。这类硬件需要更大的片上存储、跨 CTA 调度与同步资源，数据中心卡通常有更高的 SM/TMEM/带宽预算和更稳定的长期占用，因此更常提供；5090 不支持 2-CTA MMA。

## Module 4：完整 GEMM

统一形状 4096³、bf16、tile=128×64×64。所有性能栏实测后填写。

| 实现 | TFLOPS | 对 cuBLAS 达成率 | 时间主要花在哪 |
| --- | ---: | ---: | --- |
| naive（assignment01，fp32） | | | |
| 4.1 tiled | 29.5 | 3% | 逐元素 global→shared staging 及其 long-scoreboard 依赖 |
| 4.2 TMA | 340.3 | 33% | 单缓冲下等待 mma 消费完成（empty barrier） |
| 4.3 pipeline（S=3） | 257.9 | 24% | 多级 shared buffer 降低 block 驻留数，CTA 间延迟隐藏能力下降 |
| cuBLAS | 约 1040 | 100% | Tensor Core 主路径 |

4.1 与 4.2 在同一个 Slurm allocation 中依次运行；两者均为 `M=N=K=4096` 且判测 PASS。

### 4.1 tiled

实现 grid 覆盖所有输出 tile，K 维循环中用 `st.shared` + swizzle staging，mma 结果在 TMEM 中累加。

实测 `4.65 ms / 29.5 TFLOPS`，同轮 cuBLAS 为 `1035.2 TFLOPS`，达成率约 `3%`。结合机器平衡点，该版本主要受普通 CUDA 指令执行的 staging 和同步限制，并未打满 HBM 或 Tensor Core。

Nsight Compute source report 显示，最高采样和绝大多数 long-scoreboard stall 集中在第 112、118 行的 A/B staging store（`ST.E.U16`）。4.1 的 staging 成本包括逐元素 global 地址计算、global load、`swz128` 地址计算、shared store、循环控制、proxy fence 和 CTA 同步。报告中 `SM Busy=24.18%`、`No Eligible=75.66%`、`DRAM Throughput=0.31%`；`BAR.SYNC` 和 `LDTM.x8` 有一定等待，但不是主导因素。

### 4.2 TMA

TMA 将二维地址生成、global load、swizzled shared store 和逐元素循环从普通 CUDA 指令转交给硬件，由 `CUtensorMap`、`cp.async.bulk.tensor` 和 mbarrier `expect_tx` 描述搬运。实测 `0.40 ms / 340.3 TFLOPS`，同轮 cuBLAS 为 `1040.0 TFLOPS`，达成率约 `33%`，相对 4.1 提升约 `11.5×`。

Nsight Compute 显示 `SM Busy=42.32%`、Tensor pipeline 利用率 `26.27%`、`DRAM Throughput=4.45%`、`No Eligible=86.54%`。stall samples 中 long scoreboard 为 `6554/11355≈58%`，barrier 为 `1244/11355≈11%`。最大热点是第 192 行 `mbar_wait(mbar_empty_u32, phase_empty)` 对应的等待循环（约 4042 samples），说明单缓冲主要等待本轮 mma 消费完 shared memory，才能让下一轮 TMA 覆写。等待 TMA 数据到达的 full barrier 只有约 26 个聚合 samples，并非主要瓶颈；第 154 行 `__syncthreads()` 有 839 samples，其中 830 为 barrier stall，有可见但次要的开销。

| 指标/时间段 | 4.1 tiled | 4.2 TMA |
| --- | ---: | ---: |
| 正常运行时间 | 4.65 ms | 0.40 ms |
| SM Busy | 24.18% | 42.32% |
| No Eligible | 75.66% | 86.54% |
| DRAM Throughput | 0.31% | 4.45% |
| 主要 source hotspot | A/B `ST.E.U16` staging | `mbar_wait(empty)` |

### 4.3 多级流水

在同一个 Slurm allocation 中运行 `sweep_stages.sh`，所有配置均严格对拍
PASS。`4096³` 的 S=3 用时 `0.53 ms`，性能为 `257.9 TFLOPS`，同轮
cuBLAS 为 `1059.8 TFLOPS`，达成率为 `24%`。

| 形状 | S=2 | S=3 | S=4 | S=6 |
| --- | ---: | ---: | ---: | ---: |
| 4096³ | 294.3 | 257.9 | 209.0 | 121.8 |
| 256×4096×16384 | 114.4 | 115.2 | 114.5 | 114.4 |

表中单位为 TFLOPS。每个 stage 保存一个 `128×64` 的 A tile 和一个
`64×64` 的 B tile，因此每 stage 的数据区为

$$
(BM+BN)BK\times 2=(128+64)\times64\times2=24576\ \text{B}.
$$

kernel 还为 1024 B 对齐余量申请动态 shared memory。Nsight Compute 的
LaunchStats/Occupancy 结果如下：

| S | 动态 shared/block | shared 限制的 blocks/SM | 理论 occupancy |
| ---: | ---: | ---: | ---: |
| 2 | 50176 B | 4 | 25.00% |
| 3 | 74752 B | 3 | 18.75% |
| 4 | 99328 B | 2 | 12.50% |
| 6 | 148480 B | 1 | 6.25% |

`4096³` 有 `(4096/128)×(4096/64)=2048` 个 CTA，grid 足够大，多个
resident CTA 本身可以隐藏 TMA、mbarrier 和 mma 的延迟。S 从 2 增至 6
时，每 SM 可驻留 CTA 数从 4 降至 1，损失的 CTA 间并发大于更深预取带来
的收益，因此性能单调下降。这里不能得出“stage 越多越快”的结论。

`256×4096×16384` 只有 `2×64=128` 个 CTA，但每个 CTA 有 256 个 K tile。
这个 grid 在该 GPU 上本来就难以让每个 SM 同时驻留多个 CTA，所以增大 S
几乎没有进一步损失 block 间并发；与此同时 S=2 已足以覆盖当前 TMA/mma
流水的主要延迟，S=3、4、6 也没有可测的额外收益，四个结果都在
`114--115 TFLOPS` 的波动范围内。

选择 S=3。每个 stage 内必须先 TMA 写满，再由 mma 读取，mma 完成并通知
`empty[s]` 后该 stage 才能被下一轮 TMA 覆写；重叠发生在不同 stage 之间：

```text
时间片        t0       t1       t2       t3       t4       t5
stage 0     TMA k0   MMA k0                    TMA k3   MMA k3
stage 1              TMA k1   MMA k1                    TMA k4
stage 2                       TMA k2   MMA k2                    ...

跨 stage 重叠          TMA k1 与 MMA k0
                      TMA k2 与 MMA k1
                      TMA k3 与 MMA k2
```

图是依赖关系示意，不表示各操作等时长。`full[s]` 保证 TMA→mma，
`empty[s]` 保证 mma→下一次复用该 stage 的 TMA。

**(a) 瓶颈变化。** 4.1 的热点是普通线程执行的 global load、swizzle
地址计算和 shared store，主要表现为 staging 指令及 long-scoreboard 等待。
4.2 用 TMA 消掉这批普通 CUDA 指令后，单缓冲的 `empty` 依赖使下一轮 TMA
必须等待当前 mma 消费完 shared tile。4.3 在语义上解除了这个串行依赖，
让不同 stage 的 TMA 和 mma 重叠；但当前 tile/grid 上，新增 shared memory
降低 occupancy 的代价更大。S=3 只有 257.9 TFLOPS，低于 4.2 的
340.3 TFLOPS，因此瓶颈并没有转成“Tensor Core 已打满”，而是转成了
shared-memory 容量造成的低 CTA/warp 并发，以及仍存在的 barrier 和串行
producer/consumer 控制开销。

**(b) 梯子逐级归因。** assignment01 naive 每个输出元素直接从 global
读取输入并用普通 fp32 运算；4.1 用 shared tile 复用 A/B、用 TMEM 保存跨 K
累加器并使用 Tensor Core，减少重复 global 流量。4.1→4.2 把二维地址生成、
global load、swizzle 和 shared store 从普通线程指令卸载给 TMA。4.2→4.3
增加循环缓冲，减少“本轮 mma 完成后才能发下一轮 TMA”的暴露延迟，但以
每 stage 24576 B shared memory 和更多 mbarrier 状态为代价；本次实测该
交换在大 grid 上是负收益。

**(c) 容量限制。** 固定 `BM=128, BN=64, BK=64` 只增加 stage 时，TMEM
中的输出累加器不随 S 增长，每 CTA 始终占 `128×64×4=32768 B`；shared
memory 却每增加一级增长 24576 B，所以首先限制 residency 的一定是 shared
memory，S=6 已只允许 1 block/SM。若继续扩大输出 tile，TMEM 也会随
`BM×BN` 增长并最终成为硬上限。3.4 的 `cta_group::2` 将每 CTA 的 B staging
减半，能把省下的 shared memory 用于更多 stage 或更大 tile，但不会减少
输出 accumulator 所需的 TMEM；因此它只能推迟 shared-memory 上限，不能
消除后续的 TMEM 容量限制。

### 4.5 skinny GEMM

本次实验在独占的 B300 allocation 中运行，使用 Module 0 的 dense bf16
计算峰值 2500 TFLOPS 和 HBM 带宽 8000 GB/s。机器平衡点为

$$
I_{\rm balance}=\frac{2500\times1000}{8000}
=312.5\ \mathrm{FLOP/byte}.
$$

表中的 compute roof 达成率等于实测 TFLOPS 除以 2500，memory roof
达成率等于实测 TFLOPS 除以 $8AI$。程序中的 GB/s 按题面公式计算逻辑
字节量，属于有效带宽。在同一权重上重复计时时，L2 命中会使实际 HBM
流量小于逻辑字节量，因此它不能代替 Nsight Compute 的 DRAM 事务指标。

固定的七组投影形状如下。

| 投影层 | N | K |
| --- | ---: | ---: |
| f_b_proj | 1536 | 128 |
| q_b_proj | 2304 | 1536 |
| o_proj | 7168 | 1536 |
| fused_qkv_a_proj | 2112 | 7168 |
| in_proj_qkvgfab | 6288 | 7168 |
| dense_down_proj | 7168 | 8448 |
| dense_gate_up_proj | 16896 | 7168 |

在离散采样点中，除 f_b_proj 外的六组形状都在 M <= 256 时满足
$AI<312.5$，理论 memory roof 较低；从 M=1024 起满足
$AI>312.5$，理论 compute roof 较低。f_b_proj 的最大 AI 约为
118，全程位于机器平衡点以下。

#### f_b_proj（N=1536, K=128）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 0.1 | 98.0 | 0.0% | 1.2% |
| 8 | 7.5 | 0.5 | 68.2 | 0.0% | 0.9% |
| 16 | 14.1 | 1.6 | 111.0 | 0.1% | 1.4% |
| 64 | 41.5 | 6.3 | 152.0 | 0.3% | 1.9% |
| 256 | 80.8 | 25.4 | 314.2 | 1.0% | 3.9% |
| 1024 | 105.9 | 86.2 | 814.1 | 3.4% | 10.2% |
| 4096 | 114.8 | 198.0 | 1724.3 | 7.9% | 21.6% |
| 16384 | 117.3 | 331.8 | 2828.8 | 13.3% | 35.4% |
| 65536 | 117.9 | 410.5 | 3480.3 | 16.4% | 43.5% |

#### q_b_proj（N=2304, K=1536）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 0.6 | 572.3 | 0.0% | 7.2% |
| 8 | 7.9 | 7.0 | 887.5 | 0.3% | 11.1% |
| 16 | 15.7 | 20.0 | 1272.7 | 0.8% | 15.9% |
| 64 | 59.8 | 78.9 | 1318.8 | 3.2% | 16.5% |
| 256 | 200.3 | 293.6 | 1465.4 | 11.7% | 18.3% |
| 1024 | 485.1 | 717.6 | 1479.4 | 28.7% | 18.5% |
| 4096 | 752.3 | 1013.2 | 1346.7 | 40.5% | 16.8% |
| 16384 | 872.5 | 1177.1 | 1349.1 | 47.1% | 16.9% |
| 65536 | 908.8 | 1262.3 | 1389.0 | 50.5% | 17.4% |

#### o_proj（N=7168, K=1536）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 3.3 | 3319.1 | 0.1% | 41.5% |
| 8 | 7.9 | 31.0 | 3902.7 | 1.2% | 48.8% |
| 16 | 15.8 | 60.7 | 3843.8 | 2.4% | 48.0% |
| 64 | 60.9 | 192.5 | 3160.8 | 7.7% | 39.5% |
| 256 | 212.9 | 562.5 | 2642.0 | 22.5% | 33.0% |
| 1024 | 565.9 | 784.7 | 1386.6 | 31.4% | 17.3% |
| 4096 | 966.5 | 1179.1 | 1220.0 | 47.2% | 15.2% |
| 16384 | 1174.3 | 1251.8 | 1066.0 | 50.1% | 13.3% |
| 65536 | 1241.0 | 1304.6 | 1051.2 | 52.2% | 13.1% |

#### fused_qkv_a_proj（N=2112, K=7168）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 3.0 | 2953.9 | 0.1% | 36.9% |
| 8 | 8.0 | 23.5 | 2946.1 | 0.9% | 36.8% |
| 16 | 15.8 | 45.6 | 2880.5 | 1.8% | 36.0% |
| 64 | 61.6 | 135.4 | 2199.4 | 5.4% | 27.5% |
| 256 | 221.3 | 446.8 | 2019.3 | 17.9% | 25.2% |
| 1024 | 629.1 | 871.5 | 1385.2 | 34.9% | 17.3% |
| 4096 | 1166.7 | 1101.0 | 943.7 | 44.0% | 11.8% |
| 16384 | 1483.6 | 1149.6 | 774.8 | 46.0% | 9.7% |
| 65536 | 1591.7 | 1299.8 | 816.6 | 52.0% | 10.2% |

#### in_proj_qkvgfab（N=6288, K=7168）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 4.6 | 4567.2 | 0.2% | 57.1% |
| 8 | 8.0 | 37.8 | 4736.2 | 1.5% | 59.2% |
| 16 | 15.9 | 71.8 | 4506.9 | 2.9% | 56.3% |
| 64 | 62.8 | 309.6 | 4930.4 | 12.4% | 61.6% |
| 256 | 237.8 | 852.2 | 3583.1 | 34.1% | 44.8% |
| 1024 | 784.2 | 1150.0 | 1466.4 | 46.0% | 18.3% |
| 4096 | 1842.7 | 1140.4 | 618.9 | 45.6% | 7.7% |
| 16384 | 2781.0 | 1269.9 | 456.6 | 50.8% | 5.7% |
| 65536 | 3186.7 | 1292.9 | 405.7 | 51.7% | 5.1% |

#### dense_down_proj（N=7168, K=8448）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 4.4 | 4449.8 | 0.2% | 55.6% |
| 8 | 8.0 | 37.5 | 4690.9 | 1.5% | 58.6% |
| 16 | 15.9 | 72.7 | 4564.0 | 2.9% | 57.0% |
| 64 | 63.0 | 256.4 | 4073.1 | 10.3% | 50.9% |
| 256 | 240.1 | 885.4 | 3687.0 | 35.4% | 46.1% |
| 1024 | 810.1 | 929.4 | 1147.3 | 37.2% | 14.3% |
| 4096 | 1991.9 | 1258.4 | 631.7 | 50.3% | 7.9% |
| 16384 | 3135.6 | 1278.1 | 407.6 | 51.1% | 5.1% |
| 65536 | 3661.1 | 1310.1 | 357.8 | 52.4% | 4.5% |

#### dense_gate_up_proj（N=16896, K=7168）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 5.9 | 5859.0 | 0.2% | 73.2% |
| 8 | 8.0 | 50.0 | 6266.1 | 2.0% | 78.3% |
| 16 | 15.9 | 97.9 | 6137.9 | 3.9% | 76.7% |
| 64 | 63.2 | 367.2 | 5810.0 | 14.7% | 72.6% |
| 256 | 243.6 | 964.2 | 3957.8 | 38.6% | 49.5% |
| 1024 | 850.9 | 1127.3 | 1324.9 | 45.1% | 16.6% |
| 4096 | 2258.2 | 1157.1 | 512.4 | 46.3% | 6.4% |
| 16384 | 3850.2 | 1302.4 | 338.3 | 52.1% | 4.2% |
| 65536 | 4673.9 | 1310.1 | 280.3 | 52.4% | 3.5% |

#### 实验结论

**(a) M 方向的性能变化。**

对六个 K >= 1536 的常规投影，M <= 16 时仅达到计算峰值的
0.0% 到 3.9%。M=64 时仍只有 3.2% 到 14.7%，到 M=256
才升至 11.7% 到 38.6%。性能在 M=1024 以下已经明显偏离大 M
平台，M <= 64 时下降最剧烈。

M=4096 后多数形状达到 1.10 到 1.26 PFLOPS，M=16384 到
65536 基本稳定在 1.15 到 1.31 PFLOPS，相当于 B300 dense bf16
计算峰值的 46% 到 52%。f_b_proj 是例外，它到 M=65536
也只有 410.5 TFLOPS。

**(b) 两种 roof 的比较。**

按 $AI<312.5$ 的判断，七个投影在 M <= 256 的采样点都属于理论
memory-bound 区域。六个常规投影从 M=1024 起进入理论
compute-bound 区域，f_b_proj 全程留在 memory-bound 一侧。

实测 memory-roof 达成率还能区分带宽限制的强弱。M <= 16 时，
dense_gate_up_proj 达到 73% 到 78%，带宽限制最明显；
in_proj_qkvgfab 和 dense_down_proj 约为 56% 到 59%，也有较强
带宽特征。o_proj 约为 42% 到 49%，fused_qkv_a_proj 约为
36% 到 37%，两者还受到固定开销和并行度的影响。q_b_proj 只有
7% 到 16%，f_b_proj 不超过 2%，它们虽然位于 memory-bound
理论区间，却远未用满该 roof。

M >= 1024 后，除 f_b_proj 外都是 compute roof 更低，实测在大 M
达到该 roof 的约 46% 到 52%。此时较低的 %BW 不表示带宽性能变差，
因为 memory roof 已经不再是限制更紧的那个上限。

**(c) f_b_proj 的限制。**

当 $M\to\infty$ 时，这个形状的 AI 上限为

$$
\lim_{M\to\infty} AI
=\frac{NK}{N+K}
=\frac{1536\times128}{1536+128}
\approx118.2\ \mathrm{FLOP/byte}.
$$

它仍远低于 312.5，因此 f_b_proj 无法随着 M 增大跨入
compute-bound 区域。它的 K 只有 128，每个输出 tile 可执行的 K
方向 Tensor Core 工作很少，无法充分摊薄 kernel 启动、算法选择、tile
调度和 epilogue 等固定成本。N 也只有 1536，小 M 时可并行的输出 tile
数量有限。结果是小 M 阶段约 4 到 6 微秒的固定耗时占主导，大 M 时又受
低 AI 限制，两个 roof 的达成率都不高。

**(d) vLLM 选择 skinny CUDA Core kernel 的原因。**

M <= 16 时，cuBLAS Tensor Core 路径的工作量很小，实测耗时仍有约
4 到 41 微秒。M 方向的 tile 大量空置，可同时运行的 tile 数也少，
Tensor Core staging、布局转换和 kernel 调度成本难以摊薄。专门的 skinny
CUDA Core kernel 可以采用贴合小 M 的线程映射，直接执行 FMA，并省去
通用 Tensor Core GEMM 的一部分 setup 和 tile 浪费。此时较低的理论峰值
并不妨碍它获得更短的端到端延迟，因此 vLLM 在 M <= 16 选择这条路径。

#### 完整运行输出

~~~text
layer                    M      N      K        us    TFLOPS      GB/s      AI  %TCpeak      %BW
f_b_proj                 1   1536    128       4.0       0.1      98.0     1.0     0.0%     1.2%
f_b_proj                 8   1536    128       6.2       0.5      68.2     7.5     0.0%     0.9%
f_b_proj                16   1536    128       4.0       1.6     111.0    14.1     0.1%     1.4%
f_b_proj                64   1536    128       4.0       6.3     152.0    41.5     0.3%     1.9%
f_b_proj               256   1536    128       4.0      25.4     314.2    80.8     1.0%     3.9%
f_b_proj              1024   1536    128       4.7      86.2     814.1   105.9     3.4%    10.2%
f_b_proj              4096   1536    128       8.1     198.0    1724.3   114.8     7.9%    21.6%
f_b_proj             16384   1536    128      19.4     331.8    2828.8   117.3    13.3%    35.4%
f_b_proj             65536   1536    128      62.8     410.5    3480.3   117.9    16.4%    43.5%

q_b_proj                 1   2304   1536      12.4       0.6     572.3     1.0     0.0%     7.2%
q_b_proj                 8   2304   1536       8.0       7.0     887.5     7.9     0.3%    11.1%
q_b_proj                16   2304   1536       5.7      20.0    1272.7    15.7     0.8%    15.9%
q_b_proj                64   2304   1536       5.7      78.9    1318.8    59.8     3.2%    16.5%
q_b_proj               256   2304   1536       6.2     293.6    1465.4   200.3    11.7%    18.3%
q_b_proj              1024   2304   1536      10.1     717.6    1479.4   485.1    28.7%    18.5%
q_b_proj              4096   2304   1536      28.6    1013.2    1346.7   752.3    40.5%    16.8%
q_b_proj             16384   2304   1536      98.5    1177.1    1349.1   872.5    47.1%    16.9%
q_b_proj             65536   2304   1536     367.5    1262.3    1389.0   908.8    50.5%    17.4%

o_proj                   1   7168   1536       6.6       3.3    3319.1     1.0     0.1%    41.5%
o_proj                   8   7168   1536       5.7      31.0    3902.7     7.9     1.2%    48.8%
o_proj                  16   7168   1536       5.8      60.7    3843.8    15.8     2.4%    48.0%
o_proj                  64   7168   1536       7.3     192.5    3160.8    60.9     7.7%    39.5%
o_proj                 256   7168   1536      10.0     562.5    2642.0   212.9    22.5%    33.0%
o_proj                1024   7168   1536      28.7     784.7    1386.6   565.9    31.4%    17.3%
o_proj                4096   7168   1536      76.5    1179.1    1220.0   966.5    47.2%    15.2%
o_proj               16384   7168   1536     288.2    1251.8    1066.0  1174.3    50.1%    13.3%
o_proj               65536   7168   1536    1106.2    1304.6    1051.2  1241.0    52.2%    13.1%

fused_qkv_a_proj         1   2112   7168      10.3       3.0    2953.9     1.0     0.1%    36.9%
fused_qkv_a_proj         8   2112   7168      10.3      23.5    2946.1     8.0     0.9%    36.8%
fused_qkv_a_proj        16   2112   7168      10.6      45.6    2880.5    15.8     1.8%    36.0%
fused_qkv_a_proj        64   2112   7168      14.3     135.4    2199.4    61.6     5.4%    27.5%
fused_qkv_a_proj       256   2112   7168      17.3     446.8    2019.3   221.3    17.9%    25.2%
fused_qkv_a_proj      1024   2112   7168      35.6     871.5    1385.2   629.1    34.9%    17.3%
fused_qkv_a_proj      4096   2112   7168     112.6    1101.0     943.7  1166.7    44.0%    11.8%
fused_qkv_a_proj     16384   2112   7168     431.5    1149.6     774.8  1483.6    46.0%     9.7%
fused_qkv_a_proj     65536   2112   7168    1526.6    1299.8     816.6  1591.7    52.0%    10.2%

in_proj_qkvgfab          1   6288   7168      19.7       4.6    4567.2     1.0     0.2%    57.1%
in_proj_qkvgfab          8   6288   7168      19.1      37.8    4736.2     8.0     1.5%    59.2%
in_proj_qkvgfab         16   6288   7168      20.1      71.8    4506.9    15.9     2.9%    56.3%
in_proj_qkvgfab         64   6288   7168      18.6     309.6    4930.4    62.8    12.4%    61.6%
in_proj_qkvgfab        256   6288   7168      27.1     852.2    3583.1   237.8    34.1%    44.8%
in_proj_qkvgfab       1024   6288   7168      80.3    1150.0    1466.4   784.2    46.0%    18.3%
in_proj_qkvgfab       4096   6288   7168     323.8    1140.4     618.9  1842.7    45.6%     7.7%
in_proj_qkvgfab      16384   6288   7168    1163.0    1269.9     456.6  2781.0    50.8%     5.7%
in_proj_qkvgfab      65536   6288   7168    4569.2    1292.9     405.7  3186.7    51.7%     5.1%

dense_down_proj          1   7168   8448      27.2       4.4    4449.8     1.0     0.2%    55.6%
dense_down_proj          8   7168   8448      25.9      37.5    4690.9     8.0     1.5%    58.6%
dense_down_proj         16   7168   8448      26.6      72.7    4564.0    15.9     2.9%    57.0%
dense_down_proj         64   7168   8448      30.2     256.4    4073.1    63.0    10.3%    50.9%
dense_down_proj        256   7168   8448      35.0     885.4    3687.0   240.1    35.4%    46.1%
dense_down_proj       1024   7168   8448     133.4     929.4    1147.3   810.1    37.2%    14.3%
dense_down_proj       4096   7168   8448     394.2    1258.4     631.7  1991.9    50.3%     7.9%
dense_down_proj      16384   7168   8448    1552.5    1278.1     407.6  3135.6    51.1%     5.1%
dense_down_proj      65536   7168   8448    6058.3    1310.1     357.8  3661.1    52.4%     4.5%

dense_gate_up_proj       1  16896   7168      41.4       5.9    5859.0     1.0     0.2%    73.2%
dense_gate_up_proj       8  16896   7168      38.7      50.0    6266.1     8.0     2.0%    78.3%
dense_gate_up_proj      16  16896   7168      39.6      97.9    6137.9    15.9     3.9%    76.7%
dense_gate_up_proj      64  16896   7168      42.2     367.2    5810.0    63.2    14.7%    72.6%
dense_gate_up_proj     256  16896   7168      64.3     964.2    3957.8   243.6    38.6%    49.5%
dense_gate_up_proj    1024  16896   7168     220.0    1127.3    1324.9   850.9    45.1%    16.6%
dense_gate_up_proj    4096  16896   7168     857.5    1157.1     512.4  2258.2    46.3%     6.4%
dense_gate_up_proj   16384  16896   7168    3047.2    1302.4     338.3  3850.2    52.1%     4.2%
dense_gate_up_proj   65536  16896   7168   12116.5    1310.1     280.3  4673.9    52.4%     3.5%
~~~

## Module 5：低精度与 block scaling

### 5.1 per-tensor E4M3 与 outlier

量化使用 `scale = amax / 448`。输入先除以 scale 并转换成
`torch.float8_e4m3fn`，反量化时再乘回同一个 scale。固定随机种子下的
运行结果如下。per-block 一行对每个目标值选择原张量中距离最近的元素，
这些元素可能来自不同 block。

| 采样点 x≈ | 0.5 | 0.1 | 0.01 | 0.005 | 3000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| per-tensor，含 outlier | 4.611e-2 | 4.634e-2 | 3.085e-1 | 1.000e+0 | 0.000e+0 |
| per-tensor，去掉 outlier | 3.086e-4 | 1.795e-2 | 4.795e-3 | 1.581e-2 | N/A |
| 1x128 per-block，含 outlier | 1.366e-3 | 2.656e-2 | 1.240e-3 | 1.505e-2 | 0.000e+0 |

#### (a) 去掉 outlier 后的变化

`x≈0.5` 处的相对误差从 `4.611e-2` 降到 `3.086e-4`，降至原来的
约 `0.67%`，也就是缩小约 149 倍。包含 outlier 时，整个张量的 scale
由 3000 决定，`[-1,1]` 内的普通值只能使用 E4M3 靠近零点的少量格点。

#### (b) 量化为零的阈值

E4M3FN 的最小正 subnormal 为

$$
q_{\min}=2^{-9}.
$$

转换采用 round-to-nearest-even。零和 $q_{\min}$ 的中点为
$2^{-10}$，恰好落在中点时也舍入到零。因此原始输入满足

$$
|x|\le \frac{q_{\min}}{2}\,\mathrm{scale}
=2^{-10}\,\mathrm{scale}
$$

时会量化为零。本实验含 outlier 时有

$$
\mathrm{scale}=\frac{3000}{448}\approx6.6964,
\qquad
x_{\rm zero}=2^{-10}\times6.6964\approx0.006539.
$$

这与结果一致。`x≈0.005` 被量化为零，相对误差为 1；`x≈0.01`
没有归零，但已经落入 E4M3 的低精度 subnormal 区域，相对误差达到
`30.85%`。

#### (c) 1x128 per-block scale

程序把补零后的输入重排成若干个 128 元素 block，每个 block 独立使用
自己的绝对值最大值计算 scale。outlier 位于从 0 开始编号的 `block 78`，
也就是第 79 个 block。下面的平均相对误差只统计普通元素，含 outlier 的
block 会排除数值 3000 本身。

| block | 有效普通元素 | amax | scale | 归零阈值 | 实际归零数 | 普通元素平均相对误差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 不含 outlier，block 0 | 128 | 9.975e-1 | 2.227e-3 | 2.174e-6 | 0 | 2.324e-2 |
| 含 outlier，block 78 | 16 | 3.000e3 | 6.696e0 | 6.539e-3 | 0 | 2.572e-2 |

不含 outlier 的 block 将动态范围收紧到约 `[-1,1]`，归零阈值降低到
`2.174e-6`。全张量实验中会归零的 `x≈0.005` 在 per-block 量化后只剩
约 `1.505%` 相对误差。含 outlier 的 block 仍沿用 3000 决定的大 scale，
其中绝对值不超过 `0.006539` 的普通元素仍会归零。

本次随机样本的最后一个 block 只有 16 个普通元素，它们恰好都高于该阈值，
所以实测归零数是 0。这个结果不改变阈值本身，它说明 per-block scaling
把 outlier 的影响限制在它所在的 block，没有继续影响前面的 78 个完整 block。

### 5.2 block scaling 代数

当前实现通过 `uv run pytest tests/test_block_scale.py` 的全部三项测试。

#### (a) 两种 scale 的乘回位置

row/column scale 对 A 的每一行和 B 的每一输出行各使用一个常数。令
$A_{mk}=s^A_m\hat A_{mk}$、$B_{nk}=s^B_n\hat B_{nk}$，则

$$
C_{mn}=\sum_k A_{mk}B_{nk}
=s^A_m s^B_n\sum_k\hat A_{mk}\hat B_{nk}.
$$

对固定的 $(m,n)$，$s^A_m s^B_n$ 在整个 K 归约中保持不变，所以可以
完成点积后只乘回一次。

K-block scale 每 `SEG` 个 K 元素改变一次。令 $b(k)$ 表示 k 所属的
block，并令 $A_{mk}=s^A_{m,b(k)}\hat A_{mk}$、
$B_{nk}=s^B_{n,b(k)}\hat B_{nk}$，则

$$
C_{mn}=\sum_b\left(
s^A_{m,b}s^B_{n,b}
\sum_{k\in b}\hat A_{mk}\hat B_{nk}
\right).
$$

不同 block 的 $s^A_{m,b}s^B_{n,b}$ 通常不同，每段 partial sum 必须先乘回
本段的 scale 乘积，再沿 block 维求和。错误实现先累加全部归一化 partial
sum，再统一乘第一段 scale，等价于错误地假设所有 block 的 scale 相同。

#### (b) scale 为什么与 K 归约对齐

CUTLASS 对 block-scaled GEMM 给出的逻辑布局为

$$
SFA\in\mathbb{R}^{M\times\lceil K/SV\rceil},\qquad
SFB\in\mathbb{R}^{N\times\lceil K/SV\rceil},
$$

其中 `SV` 是 16 或 32。对应的指令语义可以写成

$$
D_{ij}=C_{ij}+\sum_k
\left(A_{ik}SFA_{i,\lfloor k/SV\rfloor}\right)
\left(B_{jk}SFB_{j,\lfloor k/SV\rfloor}\right).
$$

GEMM mainloop 本来就沿 K 方向依次读取 A、B 的连续片段并形成 partial
sum。scale 采用相同的 K 分段后，每段计算期间，A 和 B 各自使用的 scale
都保持不变。Tensor Core 可以在累加该段结果前应用一次
$SFA_{i,b}SFB_{j,b}$，随后再进入下一段，无须对每个输入元素单独查找和
广播 scale，也不会把两个 scale 不同的 K 段混入同一个未恢复的归约。

物理存储还要满足硬件供数要求。CUTLASS 使用 512 B 的 scale-factor
basic block，每块组织 128 个 M 或 N 位置以及 K 方向的 4 个 scale，多个
basic block 按 K-major 排列。当前 K tile 所需的输入和 scale 因而可以按
固定模式搬运到 Tensor Core 使用的位置。这个布局同时服务连续访存、TMA
搬运和 `tcgen05.mma` 的 scale-vector 语义。

#### (c) 粒度 16 与 128 的取舍

粒度 16 为每个输出通道的连续 16 个 K 元素独立选择 scale。局部 amax
通常更贴近这 16 个值，量化步长和归零阈值随之减小。单个 outlier 最多
影响同一输出通道的 16 个 K 元素。5.1 的实验呈现了相同现象，普通 block
的归零阈值为 `2.174e-6`，包含 outlier 的 block 为 `6.539e-3`，而其他
block 不受影响。

metadata 增量需要区分两种比较口径。只改变 K 分组并保持每个输出通道
独立 scale 时，`K/16` 相对 `K/128` 需要 8 倍 scale。DeepSeek-V3 的
weight 128x128 scale 还在 128 个输出通道之间共享。对一个 $N\times K$
权重矩阵，两者的 scale 元素数近似为

$$
N_{\rm DS}=\frac{N}{128}\frac{K}{128},\qquad
N_{\rm NVFP4}=N\frac{K}{16},
$$

所以 NVFP4 相对 DeepSeek 128x128 分组约增加

$$
\frac{N_{\rm NVFP4}}{N_{\rm DS}}=128\times8=1024
$$

倍的 scale 元素，边界形状应使用 ceiling 后的精确数量。作业中的 NVFP4
scale 以一个 E4M3 字节保存，每 16 个 FP4 权重增加 1 B metadata；16 个
FP4 权重本身占 8 B，因此这部分 metadata 约等于 packed FP4 权重容量的
12.5%，尚未计入更高层 scale 和布局 padding。

更细粒度也要求 mainloop 更频繁地取得新 scale。kernel 需要处理更多
scale 的全局内存搬运、swizzle 地址和对齐，并把对应 scale 按正确时刻送入
Tensor Core。Blackwell 的 block-scaled MMA 和专用 scale-factor 布局承担
了其中一部分工作，但 metadata 带宽与供数复杂度仍是更高精度带来的成本。

参考资料为 [CUTLASS Blackwell SM100 GEMM 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)。

### 5.3 NVFP4

E2M1 格点为 `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`，bit3 为符号。RN-even 中点归属：0.25→0，0.75→1.0（码 2），1.25→1.0，1.75→2.0，2.5→2.0，3.5→4.0，5.0→4.0；大于 6 饱和到 6。负数先取绝对值编码，再置符号位；保留 `-0` 的语义需与硬件判测一致。

| 项目 | 判测输出 |
| --- | --- |
| `03a_encode_check` | |
| `03b_nvfp4_quant` | 三组形状均 `PASS(bad=0)` |
| `test_fp4_gemm`（maxrel） | |

SF 布局使用 `sf_swizzled_offset`，每 16 个 K 元素一个组，按 M/ K tile、`outerM/innerM/innerK` 交错存储。

#### 5.3(c) ceiling probe

普通计时取两者共有的 $M=4096,K=7168$ 形状。有效流量按每个输入元素
$2+0.5+1/16=2.5625$ B，即读 bf16、写 packed E2M1 和写 E4M3 SF
计算：

| kernel | 时间 | 有效带宽 |
| --- | ---: | ---: |
| ceiling probe | 61.28 us | 1228 GB/s |
| quant kernel | 61.48 us | 1224 GB/s |
| quant / ceiling | 1.003x（时间） | 99.7%（带宽） |

因此 quant 距离这个同形访存 ceiling 约 $1-1224/1228=0.3\%$。另外两个
probe 形状 $16384\times4096$ 和 $16384\times8192$ 分别达到 1282 GB/s
和 1307 GB/s。本组普通计时与 5.4 在同一个 GPU allocation 中重测；NCU
百分比来自相同 launch 配置的独立 profiler 运行。

用 Nsight Compute 填写 SM 利用率、访存吞吐和转换指令数，并判断剩余差距来自访存还是计算：

| NCU 指标（4096x7168） | ceiling probe | quant kernel |
| --- | ---: | ---: |
| L1/TEX throughput | 97.33% | 96.51% |
| DRAM throughput | 12.09% | 12.20% |
| Compute (SM) throughput | 14.15% | 25.61% |
| Achieved occupancy | 24.61% | 24.59% |
| executed conversion thread instructions | 151552 | 22020096 |

probe 直接取得 bf16 的 16-bit 原始存储并用 XOR 折叠，没有 bf16/fp4 数值
转换；其中剩余的少量 conversion 指令来自循环索引和地址计算。quant 的
conversion 数量显著增加，SM throughput 也高出约 11.5 个百分点，但正常
计时只比 probe 慢 0.3%，说明硬件 E2M1/E4M3 转换大部分被访存等待隐藏。

两者的 L1/TEX throughput 都已接近 97%，warp stall 分析也以等待 L1TEX
结果的 long scoreboard 为主，而 DRAM throughput 只有约 12%。因此这里
接近的是当前标量 load/store 形状的 L1/TEX 供数上限，并非 B300 的 HBM
峰值；剩余瓶颈主要来自访存指令与 L1/TEX 路径，量化计算仅造成约 0.9%
的可测额外耗时。两边都使用 `sms * 4` 个 128-thread block，所以 occupancy
约为 25%，比值口径保持一致。NCU metric replay 会扰动 kernel 时间，因此
表中的时间和 GB/s 使用未挂 profiler 的普通计时，NCU 只用于判断瓶颈。

### 5.4 融合 rms_norm + NVFP4

理论两步约 6.56 B/elem、融合约 2.56 B/elem，理想加速上限约 2.56×。
实验在同一个 `NVIDIA B300 SXM6 AC` allocation 中完成，运行时为 P0，采样到
SM/memory clock 为 1095/3996 MHz。普通计时不挂 profiler。

#### 公平基线调优

扫描 `BLOCK={128,256,512}` 和 `grid cap={1,2,4,8}*SM`。RMS、独立 quant
和 fused 分开计时并各自选配置，不用 fused 的配置约束 baseline。最终选择为：

- 独立 quant：128 threads，grid 为 `min(requiredBlocks, 4*SM)`；
- RMS：`M<1024` 使用 512 threads/一 CTA 每行；大 M 使用 `8*SM` grid，
  `K=7168` 取 256 threads，其余 K 取 128 threads；
- fused：`M<1024` 使用 512 threads/一 CTA 每行，大 M 使用 128 threads、
  grid 为 `min(M, 8*SM)`。

下面是同进程背靠背测得的默认两步配置（RMS 与 quant 均为
`512 threads, 2*SM`）和调优后两步配置。`improve` 只衡量 baseline 自身，
不含 fused：

| M | K | 默认两步/us | 调优两步/us | baseline improve |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 10.39 | 9.23 | 1.13× |
| 16 | 4096 | 11.28 | 9.22 | 1.22× |
| 256 | 4096 | 12.43 | 12.30 | 1.01× |
| 1024 | 4096 | 26.37 | 21.81 | 1.21× |
| 4096 | 4096 | 71.64 | 55.89 | 1.28× |
| 16384 | 4096 | 270.73 | 203.52 | 1.33× |
| 4096 | 7168 | 117.20 | 93.80 | 1.25× |
| 16384 | 7168 | 433.00 | 343.96 | 1.26× |
| 4096 | 8192 | 127.39 | 104.35 | 1.22× |
| 16384 | 8192 | 477.26 | 380.29 | 1.25× |

大形状的 baseline 因独立 quant 改用 128-thread CTA、RMS 增加 CTA 数量而
缩短约 18%--25%；`16384x4096` 缩短 25%，加速 1.33×。最终配置重新独立
运行一次，结果如下。最后一列定义为相同逻辑输出字节口径下的有效吞吐比
`BW_fused/BW_probe=t_probe/t_fused`：

| M | K | 调优两步/us | fused/us | 加速比 | bad bytes | fused/probe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 9.23 | 6.16 | 1.50× | 0 | 67.5% |
| 16 | 4096 | 9.22 | 6.16 | 1.50× | 0 | 67.5% |
| 256 | 4096 | 12.30 | 6.27 | 1.96× | 0 | 99.0% |
| 1024 | 4096 | 21.52 | 10.31 | 2.09× | 0 | 122.6% |
| 4096 | 4096 | 56.25 | 26.86 | 2.09× | 1 | 137.5% |
| 16384 | 4096 | 202.40 | 98.57 | 2.05× | 2 | 136.1% |
| 4096 | 7168 | 94.34 | 43.09 | 2.19× | 3 | 142.2% |
| 16384 | 7168 | 343.71 | 164.64 | 2.09× | 7 | 140.4% |
| 4096 | 8192 | 104.45 | 47.26 | 2.21× | 7 | 146.7% |
| 16384 | 8192 | 379.98 | 177.41 | 2.14× | 6 | 148.3% |

全部形状通过判测。少量 bad 来自 GPU FP32 树形 sumsq 与 host double 顺序
归约不同，比例远低于允许的 $10^{-4}$。编译器资源报告为 RMS 32、quant 37、
fused 48 registers/thread，三个 kernel 均为 0 spill，因此 local-memory spill
不是当前差距来源。

#### 分段归因

小 M 时数据量不足以形成带宽压力。`M=1/16` 的 probe 均约 4.16 us，fused
均为 6.16 us，而两步约 9.2 us；时间几乎不随 M 变化，主要由 kernel
启动/调度和单行 reduction 延迟决定。融合少一次 kernel launch，得到 1.50×，
但 RMS reduction 令 fused 仍高于纯访存 probe。`M=256` 时 fused 已接近 probe，
两步仍支付两个 kernel 的固定成本，因此加速接近 2×。

中等 M 的证据是 `M=1024,K=4096`：把 row grid 从 `2*SM` 增至最多
`8*SM` 后，baseline 从 26.37 us 降至 21.81 us；fused 为 10.31 us，达到
2.09×。此时已有足够多独立行覆盖 SM，主要收益转为删除 bf16 `mid` 的写回、
重读和第二次 kernel 调度。

大 M 的实测加速为 2.05×--2.21×，达到 2.56×字节模型的约 80%--86%。
剩余差距来自字节模型没有计入 reduction、`rsqrt`、权重读取和可能的第二次
`x` cache 访问；另一方面，两步方案刚写出的 `mid` 可部分命中 L2，因此其
实际 HBM 代价低于“完整写 2 B 再完整读 2 B”的理想化估计。5.3(c) 的 NCU
还显示当前 quant/probe 的 L1/TEX throughput 约 97%、DRAM 仅约 12%，说明
这些实现首先受标量访存指令和 L1/TEX 路径限制，不是单纯按 HBM 字节数缩放。

大 M 的 `fused/probe` 超过 100% 不表示超过物理带宽上限。5.3(c) probe 为了
匹配原始 quant，使用逐元素 16-bit load/store；fused 的第二阶段使用两次
`float4` 向量 load，并且 CTA/row 映射不同。因此这个 probe 是 5.3 quant
访存形状的 ceiling，不是 fused 的严格同形 ceiling。超过 100% 本身就是证据，
说明二者不能只用逻辑 `2.5625 B/elem` 当作相同物理访存效率来解释。

### 5.5 W4A16/Marlin 与 NVFP4

* (a) W4A16 + Marlin 是**存储量化**：权重以 int4 保存，计算时反量化到 fp16；NVFP4 是**计算量化**，Tensor Core 直接消费 fp4 及 block scale。
* (b) W4A16 主要减少权重显存容量和权重带宽，代价是反量化/重排；NVFP4 同时减少存储和数据搬运，并利用更高的低精度计算吞吐。
* (c) 小 batch decode 常由权重读取受限，因此 W4A16 的带宽/容量节省通常更直接；若硬件原生支持 NVFP4 且形状、scale 供数充分，NVFP4 还能进一步提高计算吞吐，需用 4.5 的形状实测确认。

## Module 6：TileLang 对照

### 6.1 lowering 对照实验

#### 实验环境与配置

本实验复用 `assignment01/kernels/tilelang_matmul.py` 中的 `make_matmul`。先按
题目要求只做 lowering 和设备代码编译，再申请 B300 做一次额外的
runtime 正确性对拍。环境为 TileLang `0.1.13`、apache-tvm-ffi `0.1.12`、
CUDA `13.0`（nvcc `V13.0.88`）。两个指定 target 使用完全相同的源函数和参数：

```text
M=N=256, K=128
BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
threads=128, num_stages=3
dtype=float16, accum_dtype=float32
```

复现脚本位于 `assignment02/cuda/m6_tilelang/lower_tilelang_matmul.py`，所有生成物
统一写入 `assignment02/cuda/m6_tilelang/output/`。

```bash
cd assignment02
python3 cuda/m6_tilelang/lower_tilelang_matmul.py
```

`LowerTileOp` 会读取 `Target.current()`，所以脚本在传入 target 的同时显式进入
`with tvm.target.Target(...)` 上下文。脚本调用
`tilelang.lower(..., enable_device_compile=True)`，不是只打印未编译的 IR。

#### 编译过程与输出

两次编译均成功，输出摘要如下：

```text
TileLang version: 0.1.13
Kernel: assignment01/kernels/tilelang_matmul.py::make_matmul
Shape: M=N=256, K=128
Config: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, threads=128, num_stages=3

[sm_90a]
device compilation: PASS
host IR lines: 87
device IR lines: 70
CUDA source lines: 92

[sm_100a]
device compilation: PASS
host IR lines: 87
device IR lines: 64
CUDA source lines: 92
```

完整编译输出保存在 `cuda/m6_tilelang/output/lowering.log`，GPU 申请和运行摘要
保存在 `cuda/m6_tilelang/output/gpu_run.log`。生成文件均位于同一个 `output/`
目录，包括
`matmul_sm_90a.{host.tir,device.tir,cu}` 和
`matmul_sm_100a.{host.tir,device.tir,cu}`。

#### lowering 结果对照

| 项目 | `sm_90a` | `sm_100a` |
| --- | --- | --- |
| 选中的 Tensor Core 指令 | `T.ptx_wgmma_ss("m64n128k16", ...)`，CUDA 中为 `tl::wgmma_ss<...,64,128,16>`，最终对应 `wgmma.mma_async` | `T.ptx_mma("m16n8k16", ...)`，CUDA 中为 `tl::mma_sync<...,16,8,16>`；该固定版本对这个普通 `T.gemm` 回退到 `mma.sync`，没有生成 `tcgen05` |
| descriptor 在哪里、由谁生成 | host IR 的 `tma_descriptor_args` 由 TileLang runtime 为 A/B 创建 `CUtensorMap`；device IR 还自动分配 `local.descriptor.wgmma`，由 `initialize_wgmma_descriptor` 编码 WGMMA 的 A/B smem descriptor | host 侧同样由 TileLang runtime 创建 A/B 的 TMA `CUtensorMap`；计算路径使用 `ldmatrix` 装入 lane fragment，因此没有 tcgen05 smem matrix descriptor |
| smem swizzle 在哪一步确定 | `LayoutInference` 根据 `T.gemm` 的 WGMMA 供数约束推导 shared layout，copy lowering 将它写入 TMA tensor map；本次 `tma_descriptor_args` 的 swizzle 枚举值为 `2`，即 `CU_TENSOR_MAP_SWIZZLE_64B` | 同样由 `LayoutInference` 和 TMA copy lowering 决定为 64 B swizzle；随后编译器生成与布局匹配的 `ldmatrix.x4` / `ldmatrix.x4.trans` 地址映射 |
| 数据由谁搬入 smem | 源码的 `T.copy` 被降为 TMA，producer warp 中的 elected thread 发出 `tl::tma_load`，用 mbarrier 管理三 stage 循环缓冲 | 同样由 TMA 搬运；`tl::tma_load` 和 mbarrier 结构与 `sm_90a` 相同 |

`sm_90a` 生成代码中的关键证据是：

```cpp
tl::initialize_wgmma_descriptor<2, 1, 32>(desc_a, A_shared);
tl::initialize_wgmma_descriptor<1, 256, 64>(desc_b, B_shared);
tl::wgmma_ss<..., 64, 128, 16, ...>(...);
```

`sm_100a` 生成代码中的关键证据是：

```cpp
tl::ptx_ldmatrix_x4(...);
tl::ptx_ldmatrix_x4_trans(...);
tl::mma_sync<..., 16, 8, 16, ...>(...);
```

这也核实了题面编者注：TileLang `0.1.13` 虽然包含 tcgen05 lowering，普通
`T.gemm` 只有在 accumulator 使用 `shared.tmem` 等 tcgen05 scope/shape
约束满足时才会选择它。assignment01 的实现用 `T.alloc_fragment` 创建
`C_local`，因此 `sm_100a` target 本身不足以触发 tcgen05，编译器选择了合法的
`mma.sync` fallback。

#### (a) DSL 自动完成的硬件相关决策

1. 根据 target、dtype、tile、线程数和 buffer scope 选择可用的 Tensor Core
   lowering。本实验实际选择为 `sm_90a -> WGMMA`、`sm_100a -> MMA.SYNC`。
2. 推导 WGMMA 的 warpgroup/fragment 映射，或 MMA.SYNC 的 lane fragment
   映射，并在后者中插入 `ldmatrix.x4` 与转置变体。
3. 推导 shared-memory swizzle 和对齐，把布局写入 TMA tensor map，并在
   `sm_90a` 设备代码中生成 WGMMA descriptor 及每个 stage 的 offset。
4. 将两个 `T.copy` 降为 TMA load，生成 tensor-map 参数、mbarrier、phase、
   `expect_tx` 和三 stage 循环缓冲。
5. 自动做 warp specialization。源程序写 `threads=128`，生成 kernel 为 256
   threads；前 128 threads 负责 TMA producer，后 128 threads 负责计算，并
   分别生成 `warpgroup_reg_dealloc<24>` 与 `warpgroup_reg_alloc<240>`。

与 M2--M4 的手写代码相比，descriptor 位域、swizzle 地址异或、lane fragment
坐标、TMA/mbarrier 指令序列和流水 phase 都不再由程序员逐项编码。但“自动选择”
只保证满足编译器规则，不表示一定选择该架构最新的指令；本次 `sm_100a` 的
`mma.sync` fallback 就是直接证据。

#### (b) 仍由程序员决定的参数

程序员仍需选择 `BLOCK_M/N/K`、输入和累加 dtype、转置方向、逻辑线程数、
`num_stages`、grid/tile 映射、边界策略和 epilogue。内存层次也会改变可选指令：
如果明确需要 tcgen05，需要采用满足约束的 TMEM accumulator 或显式
`T.tcgen05_gemm` 路径，而不能只把 target 改成 `sm_100a`。这些参数共同决定
shared memory/TMEM/寄存器占用、occupancy、数据复用和流水效果，最终仍需在目标
GPU 上做正确性与性能验证。

#### assignment01 7.5 “谁负责”表新增行

| 谁负责 | CUDA SIMT | cuTile | Triton | TileLang |
| --- | --- | --- | --- | --- |
| Tensor Core 指令选择与供数布局 | 用户选择并手写 MMA/WGMMA/tcgen05 指令、fragment/descriptor、smem layout 和搬运同步 | 用户表达 tile 运算与 tile shape，编译器按 target 选择指令并生成供数布局 | 用户写 `tl.dot` 并设置 block/warps/stages，编译器按 target 和 layout lower 到 Tensor Core 与数据搬运 | 用户决定 tile、dtype、threads、stages 和 memory scope；编译器在约束允许时选择 MMA/WGMMA/tcgen05，并生成 fragment/descriptor、swizzle、TMA 和同步 |
