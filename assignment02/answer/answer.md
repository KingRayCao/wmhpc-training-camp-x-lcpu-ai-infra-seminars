# 作业 2：Tensor Core & Pipeline 答案

## Module 0：环境与峰值

### 0.1 最小 mma

运行命令：`cd assignment02/cuda && make run/m0_env/01_first_mma`。

| 项目 | 结果 |
| --- | --- |
| 正确 ARCH 的输出 | `D[0][0]=2 D[0][7]=2 D[15][0]=2 D[15][7]=2` |
| 错误 ARCH 的编译/运行现象 | `CUDA error cudaErrorNoKernelImageForDevice at m0_env/01_first_mma.cu:76: no kernel image is available for execution on the device` |

解释：fatbin 中只有为目标架构生成的 cubin 可以直接执行；若存在兼容 PTX，驱动可能在运行时 JIT 到当前 GPU。目标架构不匹配且没有可用的兼容 cubin/PTX 时会在编译期或运行期报“不支持该架构/无可执行 image”等错误。

### 0.2 Tensor Core 理论峰值（推导）

本文统一采用 **dense、boost 频率、FMA=2 FLOP** 的口径。NVIDIA 产品页经常给出包含结构化稀疏的 AI TOPS/PFLOPS，下面先换算成 dense，再做比较。

理论峰值的通式为

$$
P=F_{\rm SM}\times N_{\rm SM}\times f,
$$

其中 $F_{\rm SM}$ 是每 SM 每周期的 Tensor Core FLOP，$N_{\rm SM}$ 是 SM 数，$f$ 是 boost 时钟。

#### RTX 5090

官方规格给出 21760 个 CUDA cores、2.41 GHz boost、3352 AI TOPS。Blackwell GeForce 每 SM 有 128 个 CUDA cores，因此

$$
N_{\rm SM}=21760/128=170.
$$

按课件使用的 Blackwell AI TOPS 口径，3352 AI TOPS 对应 sparse FP4。去掉结构化稀疏的 2 倍，再按 fp4 相对 bf16 的 4 倍吞吐换算：

$$
P_{\rm bf16,dense}=\frac{3352}{2\times4}=419\ \mathrm{TFLOPS}.
$$

反推每 SM 每周期吞吐：

$$
F_{\rm SM}=\frac{419\times10^{12}}{170\times2.41\times10^9}
\approx1024\ \mathrm{FLOP/cycle/SM}.
$$

用架构整数吞吐 1024 重新正向计算：

$$
P_{\rm bf16}=1024\times170\times2.41\times10^9
=419.53\ \mathrm{TFLOPS}.
$$

按 dtype 宽度关系估算：

$$
P_{\rm fp8}=2P_{\rm bf16}=839.07\ \mathrm{TFLOPS},
\qquad
P_{\rm fp4}=4P_{\rm bf16}=1678.13\ \mathrm{TFLOPS}.
$$

再给 fp4 加上结构化稀疏的 2 倍是 3356.26 TOPS，与官方 3352 AI TOPS 相差约 0.13%；差异来自产品页的时钟和 TOPS 均经过取整。

RTX 5090 FE 的 GDDR7 为 512-bit、28 Gb/s，因此理论带宽为

$$
B_{5090}=\frac{512}{8}\times28=1792\ \mathrm{GB/s}.
$$

#### B300

NVIDIA 官方 DGX B300 规格是 8 块 Blackwell Ultra SXM GPU。系统 FP8 Tensor Core 性能为 72 PFLOPS（sparse），官方脚注说明 dense 是其一半。因此单块 B300 的 dense FP8 峰值为

$$
P_{\rm fp8,dense}=\frac{72}{2\times8}=4.5\ \mathrm{PFLOPS}.
$$

按 fp8 相对 bf16 的 2 倍吞吐反推：

$$
P_{\rm bf16,dense}=4.5/2=2.25\ \mathrm{PFLOPS}.
$$

Blackwell Ultra 的架构吞吐口径为 8192 bf16 FLOP/cycle/SM；用实机 `multiProcessorCount` 和应用时钟代入通式，可以得到约 2.25 PFLOPS。DGX datasheet 没有公开产品级 SM 数和 GPU 时钟，因此不应从经过取整的 2.25 PFLOPS 反推一个伪精确时钟。

只按 dtype 宽度估算 fp4：

$$
P_{\rm fp4,estimate}=4P_{\rm bf16}=9.0\ \mathrm{PFLOPS}.
$$

但官方 DGX B300 给出的系统 FP4 是 144 PFLOPS sparse、108 PFLOPS dense，换算单卡分别为 18 和 13.5 PFLOPS。也就是说，官方 dense FP4 是简单宽度估算的 1.5 倍；这是 Blackwell Ultra 对 dense FP4 的额外增强，不是口径计算错误。

B300 单卡 HBM3e 带宽按 8 TB/s，即 8000 GB/s 计算。

#### 汇总

| 量 | RTX 5090 | B300 |
| --- | ---: | ---: |
| bf16 FLOP/cycle/SM | 1024 | 8192 |
| bf16 dense 峰值 (TFLOPS) | 419.53 | 2250 |
| fp8 dense 峰值 (TFLOPS) | 839.07 | 4500 |
| fp4 dense：按 dtype 估算 (TFLOPS) | 1678.13 | 9000 |
| fp4 dense：datasheet 对照 (TFLOPS) | 约 1676（由 3352 sparse/2） | 13500（108 PFLOPS/8） |
| datasheet 口径差异 | 3352 AI TOPS 是 rounded sparse FP4 | FP8 72 PFLOPS 是 sparse；FP4 同时列出 144 sparse / 108 dense |
| HBM/GDDR 带宽 (GB/s) | 1792 | 8000 |
| 机器平衡点 (FLOP/byte, bf16) | 234.1 | 281.25 |

机器平衡点按 dense bf16 峰值计算：

$$
I_{5090}=\frac{419.53\times1000}{1792}=234.1\ \mathrm{FLOP/byte},
$$

$$
I_{B300}=\frac{2250\times1000}{8000}=281.25\ \mathrm{FLOP/byte}.
$$

单条 m16n8k16 fp16 mma 的运算量为

$$
2MNK=2\times16\times8\times16=4096\ \mathrm{FLOP}.
$$

按 A、B 读入和 D 写回计字节数：

$$
16\times16\times2+16\times8\times2+16\times8\times4=1280\ \mathrm{byte},
$$

所以计算强度为

$$
AI_{\rm mma}=4096/1280=3.2\ \mathrm{FLOP/byte}.
$$

它比两块 GPU 的机器平衡点低约两个数量级，说明如果每次 mma 都重新从显存读取 A/B 并写回 D，Tensor Core 会严重缺数。M2--M4 的 shared-memory staging、swizzle、TMA 和 pipeline，本质上都是通过 tile 复用和搬运重叠提高整个 GEMM 的计算强度与供数效率。

官方资料位置：

* [GeForce RTX 5090 官方规格页](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)：21760 CUDA cores、3352 AI TOPS、2.41 GHz boost、32 GB GDDR7、512-bit。
* [NVIDIA DGX B300 Datasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-b300-datasheet)：可在线阅读或下载官方 PDF。
* [NVIDIA DGX B300 官方规格页](https://www.nvidia.com/en-us/data-center/dgx-b300/)：8 块 Blackwell Ultra GPU、FP4/FP8 系统性能及 sparse/dense 脚注。
* [NVIDIA Data Center GPU Line Card（PDF）](https://docs.nvidia.com/data-center-gpu/line-card.pdf)：数据中心 GPU 的单卡规格对照。
* [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)：官方说明 Blackwell Ultra 的 Tensor Core/FP4 增强。

### 0.3 概念判断

* **(a) 对。** 按 S016 口径，分子为 \(2MNK\)，分母是 A、B 读入和 D 写回的字节数。若改用只计输入、或只计 DRAM 实际事务的口径，数值会不同，必须注明。
* **(b) 对。** `mma.sync` 是 warp 协作指令，fragment 由 32 个 lane 共同提供；warp 发散执行属于未定义行为。
* **(c) 错。** 更大的形状虽提高单条指令的计算/字节比，但会增加 fragment、寄存器、指令编码和调度压力，可能降低 occupancy，且受硬件支持的 shape 限制。
* **(d) 错。** 单条 mma 的强度不等于完整 GEMM kernel 的强度；tile 复用 A/B、累加器驻留和多级缓存可显著提高整体 arithmetic intensity。

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

### 4.4 Optional

选择方向：`[a/b/c]`。实现、判测输出和每一步优化的瓶颈记录：

```text
代码/命令/结果：
```

### 4.5 skinny GEMM

对每个形状先计算
$$
AI=\frac{2MNK}{2MK+2NK+2MN},\qquad
P_{\rm mem}=AI\times B_{\rm mem}.
$$

固定 shape：

| 投影层 | N | K |
| --- | ---: | ---: |
| `f_b_proj` | 1536 | 128 |
| `q_b_proj` | 2304 | 1536 |
| `o_proj` | 7168 | 1536 |
| `fused_qkv_a_proj` | 2112 | 7168 |
| `in_proj_qkvgfab` | 6288 | 7168 |
| `dense_down_proj` | 7168 | 8448 |
| `dense_gate_up_proj` | 16896 | 7168 |

以下模板对上述每个投影层各填写一份；运行程序的完整 stdout 也应随报告保留。

#### `f_b_proj`（N=1536, K=128）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | | | | | |
| 8 | | | | | |
| 16 | | | | | |
| 64 | | | | | |
| 256 | | | | | |
| 1024 | | | | | |
| 4096 | | | | | |
| 16384 | | | | | |
| 65536 | | | | | |

#### `q_b_proj`（N=2304, K=1536）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | | | | | |
| 8 | | | | | |
| 16 | | | | | |
| 64 | | | | | |
| 256 | | | | | |
| 1024 | | | | | |
| 4096 | | | | | |
| 16384 | | | | | |
| 65536 | | | | | |

#### `o_proj`（N=7168, K=1536）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | | | | | |
| 8 | | | | | |
| 16 | | | | | |
| 64 | | | | | |
| 256 | | | | | |
| 1024 | | | | | |
| 4096 | | | | | |
| 16384 | | | | | |
| 65536 | | | | | |

#### `fused_qkv_a_proj`（N=2112, K=7168）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | | | | | |
| 8 | | | | | |
| 16 | | | | | |
| 64 | | | | | |
| 256 | | | | | |
| 1024 | | | | | |
| 4096 | | | | | |
| 16384 | | | | | |
| 65536 | | | | | |

#### `in_proj_qkvgfab`（N=6288, K=7168）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | | | | | |
| 8 | | | | | |
| 16 | | | | | |
| 64 | | | | | |
| 256 | | | | | |
| 1024 | | | | | |
| 4096 | | | | | |
| 16384 | | | | | |
| 65536 | | | | | |

#### `dense_down_proj`（N=7168, K=8448）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | | | | | |
| 8 | | | | | |
| 16 | | | | | |
| 64 | | | | | |
| 256 | | | | | |
| 1024 | | | | | |
| 4096 | | | | | |
| 16384 | | | | | |
| 65536 | | | | | |

#### `dense_gate_up_proj`（N=16896, K=7168）

| M | AI | TFLOPS | GB/s | compute roof | memory roof |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | | | | | |
| 8 | | | | | |
| 16 | | | | | |
| 64 | | | | | |
| 256 | | | | | |
| 1024 | | | | | |
| 4096 | | | | | |
| 16384 | | | | | |
| 65536 | | | | | |

结论（填实测 M 分界和平台达成率）：小 M 时 tile 不满、启动/尾块和并行度不足，Tensor Core 利用率下降；M 足够大后进入平台。若 memory-roof 达成率高而 compute-roof 达成率低，则为显存带宽受限；`f_b_proj (K=128)` 还会受极窄 K、tile 复用不足和 launch/尾部开销限制。vLLM 在 M≤16 选择 skinny CUDA Core kernel，是因为此时专用 FMA 路径的固定开销和形状利用率优于 Tensor Core tile 路径。

## Module 5：低精度与 block scaling

### 5.1 per-tensor E4M3 与 outlier

scale 定义：`scale = amax / 448`，量化为 E4M3 后再乘回 scale。

| 采样点 x≈ | 0.5 | 0.1 | 0.01 | 0.005 | 3000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 含 outlier 的相对误差 | | | | | |
| 去掉 outlier 后的相对误差 | | | | | |

回答：

* (a) 误差变化倍数：`[待填写]`。outlier 使 scale 变大，正常值落在更少的有效 E4M3 格点上。
* (b) 令 E4M3 的最小非零格点为 `q_min`，round-to-zero 的输入阈值为 \(|x|<\tfrac12 q_{min}\,scale\)。对本实验的 `float8_e4m3fn`，应把实际 dtype 的 `q_min` 与测得阈值一起报告（不要用未经验证的近似值替代）。
* (c) 1×128 block 中含 3000 的 block 会被 outlier 拉大 scale，普通元素误差/归零明显；不含 outlier 的 block 可使用接近 1/448 的 scale，精度显著改善。两类 block 的 scale、阈值和误差实测填写。

### 5.2 block scaling 代数

row/column scale：
$$
\hat A_{mk}=A_{mk}/s^A_m,\quad \hat B_{kn}=B_{kn}/s^B_n,
\quad C_{mn}=s^A_m s^B_n\sum_k\hat A_{mk}\hat B_{kn}.
$$
scale 乘积与 k 无关，所以可在完整点积外乘回一次。

K-block scale（第 b 段）：
$$
C_{mn}=\sum_b s^A_{m,b}s^B_{n,b}
\sum_{k\in b}\hat A_{mk}\hat B_{kn}.
$$
scale 随 b 改变，不能把一个因子从整个 K 归约中提出。

硬件将 scale 绑定到连续的 K=16/32 元素组，是因为 Tensor Core 的内循环本来就按 K tile 消费连续数据；scale 与该归约段对齐可在每个指令/partial sum 中直接配对，避免跨段重排和额外广播。粒度 16 相比 128 能缩小 outlier 的影响范围、降低误差；代价是 scale metadata 数量约增加 8 倍（同一 K 长度下），并增加 scale 的搬运、布局和同步复杂度。

### 5.3 NVFP4

E2M1 格点为 `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`，bit3 为符号。RN-even 中点归属：0.25→0，0.75→1.0（码 2），1.25→1.0，1.75→2.0，2.5→2.0，3.5→4.0，5.0→4.0；大于 6 饱和到 6。负数先取绝对值编码，再置符号位；保留 `-0` 的语义需与硬件判测一致。

| 项目 | 判测输出 |
| --- | --- |
| `03a_encode_check` | |
| `03b_nvfp4_quant` | |
| `test_fp4_gemm`（maxrel） | |

SF 布局使用 `sf_swizzled_offset`，每 16 个 K 元素一个组，按 M/ K tile、`outerM/innerM/innerK` 交错存储。

#### 5.3(c) ceiling probe

| kernel | GB/s |
| --- | ---: |
| ceiling probe | |
| quant kernel | |
| quant / ceiling 比值 | |

用 Nsight Compute 填写 SM 利用率、访存吞吐和转换指令数，并判断剩余差距来自访存还是计算：

```text
指标与结论：
```

### 5.4 融合 rms_norm + NVFP4

理论两步约 6.56 B/elem、融合约 2.56 B/elem，理想加速上限约 2.56×，但实际还受归约、量化转换、尾块和 M 方向并行度限制。

| M | K | 两步基线时间 | 融合时间 | 实测加速比 | 融合/5.3(c) ceiling |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | | | | |
| 16 | 4096 | | | | |
| 256 | 4096 | | | | |
| 1024 | 4096 | | | | |
| 4096 | 4096 | | | | |
| 16384 | 4096 | | | | |
| 4096 | 7168 | | | | |
| 16384 | 7168 | | | | |
| 4096 | 8192 | | | | |
| 16384 | 8192 | | | | |

性能归因（按 M 范围填写 Nsight/计算证据）：

```text
小 M：
中等 M：
大 M：
与 2.56× 理论上限的差距：
```

### 5.5 W4A16/Marlin 与 NVFP4

* (a) W4A16 + Marlin 是**存储量化**：权重以 int4 保存，计算时反量化到 fp16；NVFP4 是**计算量化**，Tensor Core 直接消费 fp4 及 block scale。
* (b) W4A16 主要减少权重显存容量和权重带宽，代价是反量化/重排；NVFP4 同时减少存储和数据搬运，并利用更高的低精度计算吞吐。
* (c) 小 batch decode 常由权重读取受限，因此 W4A16 的带宽/容量节省通常更直接；若硬件原生支持 NVFP4 且形状、scale 供数充分，NVFP4 还能进一步提高计算吞吐，需用 4.5 的形状实测确认。

## Module 6：TileLang 对照

### 6.1 lowering 对照实验

编译同一 `T.gemm` kernel，保存 CUDA lowering 和日志：

| | sm_90a | sm_100a |
| --- | --- | --- |
| 选中的 Tensor Core 指令 | | |
| descriptor 在哪里、由谁生成 | | |
| smem swizzle 在哪一步确定 | | |
| 数据由谁搬入 smem | | |

对照结论：DSL 通常自动完成目标架构指令选择、fragment/descriptor 编码、线程映射、部分 shared-memory 布局和异步搬运 lowering；程序员仍需决定 tile M/N/K、数据类型、边界策略、stages、并行/warp specialization 和 epilogue。最终以固定 TileLang 版本的 lowering 输出为证据，不能只凭源代码推断。

assignment01 7.5“谁负责”表补充：`Tensor Core 指令选择与供数布局` → **TileLang 编译器根据 target、dtype、tile 和布局约束选择/生成；程序员通过 tile、layout、stages 等参数施加约束并承担性能调优。**

## 团队选做（如完成）

| 题目 | 复现与测量 | 分析结论+证据 | 挑战实验/结果 |
| --- | --- | --- | --- |
| C1 FlashKDA→SM100 | | | |
| C2 MSA decode 专用 kernel | | | |

## 提交检查

* [ ] 所有动手题代码与判测输出已保存；FROM-SCRATCH 题有 PASS 记录。
* [ ] 所有实验表注明 GPU、CUDA/驱动版本，实测数据已填入空白栏。
* [ ] DEBUG 题包含修改前现象、修改后代码/输出和原因解释。
* [ ] GEMM/低精度实验给出性能归因，而非只列 TFLOPS。
* [ ] 团队题（若选做）包含代码、报告和答辩证据。
