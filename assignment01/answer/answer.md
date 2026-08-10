# 作业 1：非代码题答案

## Module 0：环境准备

### prob 0.2：设备参数表中的非代码部分

本题取决于实际使用的 GPU，不能从题面推导。

| 项目 | 你的卡 |
| --- | --- |
| 型号 / compute capability | NVIDIA GeForce RTX 5090/12.0 |
| SM 数量 | 170 |
| warp 大小 | 32 |
| shared memory / block | 49152 |
| 最大常驻线程 / SM | 1536 |
| 显存总量 | 33668857856 |

注意不要混淆 `sharedMemPerBlock`、`sharedMemPerBlockOptin` 和 `sharedMemPerMultiprocessor`。后面手算 occupancy 用的是每 SM 总量，而不是每 block 上限。

## Module 1：为什么要用 GPU

### prob 1.1

1. **错。** 100 TFLOPS 描述许多执行单元并行工作时的总吞吐，不描述单条指令的延迟；它不能推出 GPU 单条指令比 5 GHz CPU 更快。
2. **对。** 标称 HBM 带宽要靠大量并发且合并良好的访问才能接近。零散随机访问会产生更多内存事务，传输的字节中只有少量真正有用。
3. **对。** 严格串行依赖链没有足够并行度，新增的大量 GPU 计算单元无法同时工作；总时间主要由逐步依赖的延迟决定。
4. **错。** 1000 TFLOPS 是每秒约 \(10^{15}\) 次浮点运算的聚合吞吐。\(10^{-15}\text{s/FLOP}\) 只是吞吐的倒数，不能当作一项运算的端到端延迟。

### prob 1.2

“\(10^{12}\) FLOP / GPU 峰值吞吐”得到毫秒级，只在这些运算可大规模并行、能同时喂给大量执行单元时才有意义。严格在线算法形成宽度为 1 的依赖链：第 \(i+1\) 步必须等第 \(i\) 步完成，因此不能把 \(10^{12}\) 个操作同时发射到 GPU。

即使粗略假设每个依赖步骤只需 1 ns，\(10^{12}\) 步也约为 1000 秒。这里限制结果的是单步延迟和依赖链长度，而不是设备的总吞吐。

### prob 1.3

| 执行层次 | 软件含义 | 对应硬件 | 直接可用的存储 | 同步与通信手段 |
| --- | --- | --- | --- | --- |
| thread | kernel 的最小执行单位 | 一个 SIMT lane 上的线程状态 | 自己的寄存器；必要时使用线程私有 local memory | 自身指令顺序 |
| warp | 32 个相邻线程组成的实际调度/发射组 | warp scheduler 加一组执行 lanes | 各 lane 的寄存器；没有独立的“warp memory” | `__syncwarp`；shuffle、vote 等 warp primitives |
| block / CTA | 可协作并保证在同一 SM 上驻留的一组线程 | 一个 SM 上的 resident CTA | block 共享的 shared memory，以及各线程寄存器 | `__syncthreads`、block cooperative groups、shared memory、原子操作 |
| cluster | 显式编组、可跨 block 协作的一组 CTAs（CC 9.0+） | 同时调度在同一个 GPC 中的多个 blocks | Distributed Shared Memory，可访问 cluster 内各 block 的 shared memory | Cooperative Groups 的 `cluster.sync()`；DSM 读写和原子操作 |
| grid | 一次 kernel launch 的全部 blocks | 整个 GPU 上的多个 SM | global/constant memory，及设备级 L2 cache | 普通 launch 内无通用 grid barrier；通常以 kernel 边界同步，通过 global memory/atomics 传递结果 |

补充：cluster 只同步它包含的 blocks，不能覆盖同一 grid 中的其他 clusters。cooperative launch 配合 Cooperative Groups 可以提供受限制的 `grid.sync()`，但这也不是普通 kernel 任意 grid 都可使用的同步保证。

### prob 1.4

SIMD 把固定宽度的向量指令和单一控制流直接暴露给软件；SIMT 让程序员写许多具有独立寄存器状态和控制流语义的标量线程，硬件再把线程组织成 warp 共同发射指令。

“Volta 后每线程有独立 program counter，因此 divergence 不再有性能代价”是 **错的**。CC 7.0 起的 Independent Thread Scheduling 允许线程以更细粒度暂停、分歧和汇合，但一个 warp 仍一次发射共同指令。若 lanes 走不同分支，硬件仍需分别执行所走的路径，并屏蔽不参与该路径的 lanes，所以利用率仍会下降。

### prob 1.5

| 配置 | 耗时 (ms) | ns / 元素 |
| --- | ---: | ---: |
| CPU 单线程 | 9.006 | 2.15 ns/元素 |
| GPU `<<<1,1>>>` | 136.150 ms | 32.46 ns/元素 |
| GPU `<<<1,256>>>` | 2.325 ms | 0.55 ns/元素 |
| GPU 铺满 grid | 0.025 ms | 0.01 ns/元素 |

(a) GPU 单线程慢，是因为它只使用全卡极小的一部分资源，同时失去了 GPU 用大量 resident warps 隐藏流水线和显存延迟的能力。GPU 也没有以单线程低延迟为首要目标的复杂乱序执行、分支预测和大容量低延迟 cache 体系。

(b) 从一个 block 到铺满 grid 的加速说明 GPU 依靠的是大规模并行带来的总吞吐和延迟隐藏，而不是某个 GPU 线程本身特别快。该向量加法的算术强度很低，铺满后通常主要受显存带宽限制。

## Module 2：第一个 CUDA 程序

### prob 2.2

1. GPU 上执行、host 启动的 kernel：`__global__`
2. 只被 device 代码调用的辅助函数：`__device__`
3. host 与 device 都调用的函数：`__host__ __device__`
4. kernel 期间不变、所有线程读取的系数表：`__constant__`
5. block 内线程共享的暂存数组：`__shared__`

### prob 2.3：书面问答

(a) kernel launch 对 host 是异步的。CPU 若在 kernel 写完前读取 managed 结果，不但结果未就绪，还可能违反该平台的 managed-memory 访问规则，因此必须先建立 device 完成到 host 读取之间的同步关系。显式内存版本中的 device-to-host `cudaMemcpy` 是阻塞拷贝，它会等待同一 stream 中排在前面的 kernel，因而同步隐含在这个调用里。

(b) 两版耗时为 102.5 ms/94.5 ms。常见差异来源如下：

- 显式版本按三次大块传输搬运，行为较可预测；若 host 缓冲区是 pageable memory，运行时还可能使用内部 staging。
- managed 版本中，CPU 初始化的输入页首次被 GPU 访问时可能发生 host-to-device 缺页迁移，CPU 读取结果又可能触发 device-to-host 迁移；逐页 fault-and-migrate 有额外开销。
- `cudaMemPrefetchAsync`、访问提示、PCIe/NVLink、ATS/HMM、是否具有硬件一致性都会改变结果。在支持 host 直接访问 GPU-resident managed memory 的系统上，CPU 读取甚至不必迁移整页。

因此不能脱离 GPU、CPU-GPU 互连、OS 和驱动断言哪版一定更快。

### prob 2.4

1. **错。** 普通 kernel launch 相对 host 异步；语句返回通常只表示工作已成功提交。
2. **对。** 同一 stream 内操作按提交顺序执行；同步的 device-to-host `cudaMemcpy` 会等前面的 kernel 完成后再取得结果。
3. **错。** 非法 launch 配置等提交错误可能在启动检查处发现；kernel 内非法访存发生在异步执行阶段，通常在之后的同步、拷贝或其他返回异步错误的 CUDA API 处才暴露。

### prob 2.7：grid-stride loop 的价值与代价

价值是把“启动多少线程”和“处理多少元素”解耦：同一 kernel 可处理任意 \(n\)，可以把 block 数限制在与 SM 数相称的范围并复用线程，也便于用小 grid 调试。相邻线程在每一轮仍访问相邻元素，所以能保持合并访存。

本题只有 16384 个线程处理 \(2^{24}\) 个元素，平均每线程约处理 1024 个元素。代价包括循环、步长更新与反复判断的指令开销；更重要的是，如果 64 blocks 不足以让所有 SM 保持足够多的 resident warps，就会减少并行度和延迟隐藏。若 64 blocks 已足以占满目标 GPU，线程复用本身未必造成明显退化。

### prob 2.8

(a) block 的分配与执行先后由 GPU block scheduler 根据当时各 SM 的资源和运行状态决定；`blockIdx.x` 不规定执行顺序，device `printf` 的显示顺序也不构成调度保证。

(b) 普通程序的正确性不能依赖 block 顺序。CUDA 的 scalable programming model 要求 blocks 可按任意顺序、并行或串行执行，这样同一 grid 才能在 SM 数量不同的 GPU 上运行。跨 block 依赖通常应拆成两个 kernels，以 kernel 边界形成全局阶段同步。

## Module 3：SIMT 执行

### prob 3.1

线性编号按 x 维最快：

\[
t = x + y\,\text{blockDim.x} + z\,\text{blockDim.x}\,\text{blockDim.y}.
\]

1. \((3,5,0)\) 的线性编号为 \(3+5\times8=43\)。按 0 起始编号，它属于 warp 1（第二个 warp），lane 11（第十二个 lane）。
2. \(8\times8=64\) 个线程，共 2 个 warps。
3. 33 个线程也要占 2 个 warps。第二个 warp 只有 lane 0 有效，其余 31 lanes 在整个 kernel 中闲置；调度、指令发射和部分资源仍按 warp 粒度承担。

### prob 3.2

预测：按 warp 对齐的版本更快，理想情况下约快 2 倍；实际比值为 1.94。

奇偶分支使每个 warp 的一半 lanes 走一条路径、另一半走另一条。warp 必须执行两条路径，执行其中一条时屏蔽另一半 lanes；若两边工作量相等，warp 做了约两份路径工作。按 warp 分支时，一个 warp 只走其中一条，执行该路径时 32 lanes 都可工作。

若两分支工作量不同：

- warp 内分歧版本的单 warp 控制流成本接近两条实际路径成本之和。
- warp 对齐版本的每个 warp 只承担自己选择的那条路径；大量 warps 下总吞吐由两类 warps 的总指令工作量决定，收尾阶段还可能受较长分支的 warps 决定。不能简单理解为所有 warps 真正同时运行、时间永远只等于较长分支。

实测偏离 2 倍还会受指令调度、编译器生成代码、占用率和测量噪声影响。

### prob 3.3

(a) 写入 `buf[t]` 后没有 block barrier，各 warps 可以独立推进。线程读取 `buf[255-t]` 时，对应元素可能尚未被另一个 warp 写入，这构成跨线程的读写竞争，结果未定义。`__syncthreads()` 同时提供 block barrier 和所需的 shared-memory 可见性/顺序保证。

(b) 设线程 `t` 位于 warp \(w=\lfloor t/32\rfloor\)，其数据来源线程 `255-t` 位于 warp \(7-w\)。方程 \(w=7-w\) 没有整数解，所以每次翻转都一定跨 warp。对于一对互相读取的 warps，先运行到读取处的 warp 可能读到未写数据，而后运行的 warp 可能读到前者已经写好的数据；若调度次序较稳定，就会出现某一侧位置经常或一直正确。它依然没有正确性保证。

### prob 3.4

需要先区分同步范围：

- **block 内：**使用 `__syncthreads()`。
- **cluster 内：**CC 9.0 及以上可把若干 blocks 组成 Thread Block Cluster。cluster 内所有 blocks 会同时调度到同一个 GPC，可用 Cooperative Groups 的 `cluster.sync()` 同步，并通过 Distributed Shared Memory 访问彼此的 shared memory。
- **整个 grid：**`cluster.sync()` 不会等待其他 clusters，所以不能作为 grid-wide barrier。

cluster 内同步的核心形式如下；kernel 还必须通过 `__cluster_dims__` 或 `cudaLaunchKernelEx` 配置 cluster dimensions，否则普通 launch 对应的 cluster 只有 `1 x 1 x 1`：

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel(...) {
    cg::cluster_group cluster = cg::this_cluster();

    // cluster 内各 block 完成阶段 1
    cluster.sync();
    // 此后可安全进行依赖阶段 1 的 cluster 内操作
}
```

`cluster.sync()` 有两项保证：cluster 中所有线程到达同步点后才能继续，并且同步点前由组内线程完成的内存访问在同步点后对组内线程可见。使用 DSM 时还必须保证所有跨 block 的 DSM 访问完成后，相关 block 才能退出。

若要求**整个 grid**完成阶段 1 后才能开始阶段 2，标准做法仍是把计算拆成两个 kernels，并让第二个 kernel 在同一 stream 中排在第一个之后；kernel 边界就是 grid-wide 的阶段同步点。普通 kernel 内不存在可供任意大小 grid 使用的全局 barrier。

特殊情况下可使用 cooperative launch 和 Cooperative Groups 的 `grid.sync()`，但它要求设备支持、以 cooperative 方式启动，并受所有参与 blocks 必须可协同驻留等限制，不能替代一般做法。

因此本题最简答案是：局部跨 block 同步可用 cluster；真正覆盖整个 grid 时，标准答案仍是 kernel 边界，cooperative `grid.sync()` 只适用于满足约束的特殊场景。

### prob 3.5：两种归约的性能差异

两版加法次数相同，不代表发射的 warp 指令数和辅助指令相同。

- interleaved 版在早期每个 warp 内只有间隔分布的 lanes 活跃，许多 warp 都要参与每一轮；运行期 `% (2*s)` 还会产生额外整数运算/分支成本。
- contiguous 版把活跃线程集中在低编号连续区域。随着 `s` 减半，参与的完整 warps 很快减少，后期只需一个 warp 处理，发射效率更高。
- 两版 barrier 轮数相同，主要差异来自活跃 lanes 的分布、需要发射的 warp 数，以及索引/取模开销，而不是大 O 复杂度。

实际耗时与比值必须实测。

## Module 4：存储空间

### prob 4.1

| 空间 | 谁可见 | 生命周期 | 片上 / 片外 | 谁管理 |
| --- | --- | --- | --- | --- |
| register | 单个线程 | 线程 | 片上 | 编译器 |
| local | 单个线程 | 线程 | 逻辑私有，但物理上位于片外 device memory；可被 cache | 编译器 |
| shared | 同一 block 的线程 | block | 片上 SM | 程序员分配/使用 |
| global | grid 中所有线程，host 通过 API 可访问 | 从分配到 `cudaFree`、device reset 或进程结束 | 片外显存 | 程序员通过 Runtime/Driver API 管理 |
| constant | grid 中所有线程只读；host 可通过 API 写入 | C++ 中通常为 CUDA context | 片外 constant space，带片上 constant cache | 程序员声明和拷贝，硬件缓存 |
| L1 / L2 cache | L1 属于单个 SM；L2 为所有 SM 共享 | cache line 驻留期间，不是 CUDA 语言级对象生命周期 | 片上 | 默认由硬件/编译器/runtime 管理，可有限提示或配置 |

### prob 4.3：constant cache 的优势

constant cache 最适合一个 warp 的线程在同一条指令中读取同一个地址：一次读取可向多个 lanes 广播。若同一 warp 读取许多不同 constant 地址，请求可能按不同地址序列化，优势会消失。

本题的 8 个系数只有 32 bytes，global 版本也很容易被 L1/L2 cache 命中；同时 Horner 计算及输入/输出流量可能占据主要时间。因此即使访问模式适合 constant broadcast，两版耗时仍可能接近，实际结果为 0.0718 ms/0.0715 ms。

### prob 4.4

1. **对。** `local` 表示线程私有的逻辑作用域，物理存储位于 device/global memory 空间，而非片上寄存器。
2. **对。** 编译器若不能证明数组使用编译期常量下标，通常难以把各元素分配到静态确定的寄存器，可能把数组放到 local memory。大数组和 register spilling 也会产生 local memory。

### prob 4.6：直方图私有化为何加速

naive 版让全部线程竞争同一组 256 个 global counters，冲突的原子更新会序列化。私有化后，每个 block 先更新自己的 shared-memory histogram，把竞争范围分散到各 blocks 且使用低延迟的片上存储，最后才把每个 block 的 256 个结果合并到 global memory。

本题共有 \(2^{24}\) 个输入，naive 版约有 \(2^{24}\) 次 global atomics；私有化版最终合并只有 \(1024\times256=2^{18}\) 次 global atomics，即减少到约 1/64。shared atomics 内仍可能冲突，清零、同步和最终合并也有成本，所以实际加速比须实测。

实测性能为2.5477 ms/0.0145 ms，加速比为175.88x。

### prob 4.7：stride 与带宽

| stride | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GB/s | 1898.3 | 1190.4 | 723.4 | 613.0 | 1693.0 | 893.7 |

stride 1 时，相邻 lanes 读取相邻 `float`，一个 warp 的 128 bytes 有效数据只需少量 32-byte transactions。stride 从 1 增至 8 时，同一 warp 的地址分散到越来越多的 segments，每次事务中真正使用的字节比例下降，实测有效带宽从 1898.3 GB/s 降至 613.0 GB/s，下降约 67.7%。stride 达到 8 个 `float`（32 bytes）后，最坏可接近每 lane 一个 transaction。输出写入始终连续，因此这一段的退化主要来自输入读取不再合并。

但本实验并不是单纯改变 stride：索引是 `j = (i * stride) & (n - 1)`，且 `n` 和 stride 都是 2 的幂。stride 为 $s$ 时只会访问 $n/s$ 个不同输入元素，并在一次 kernel 中重复访问这些元素 $s$ 次。因此 stride 增大后，cache reuse、cache-set/partition 映射和请求发射效率会与 coalescing 一起影响结果；程序打印的 GB/s 又按逻辑上的“读 4 bytes + 写 4 bytes”计算，不等于实际 DRAM 传输量。

这解释了曲线为什么不必单调：stride 16 的 1693.0 GB/s 相比 stride 8 回升约 2.76 倍，而 stride 32 又降至 893.7 GB/s。仅凭这组时间无法可靠判定 16 与 32 反转的具体硬件原因；若要进一步归因，需要用 Nsight Compute 查看 L1/L2 hit rate、实际 DRAM bytes、每请求 sectors 数和 memory-partition 利用率。

### prob 4.8：occupancy

| 档位 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shared memory / block (KB) | 0.0 | 13.2 | 15.0 | 18.0 | 29.0 | 55.0 |
| 理论驻留 block / SM | 6 | 6 | 6 | 5 | 3 | 1 |
| occupancy | 100.0% | 100.0% | 100.0% | 83.3% | 50.0% | 16.7% |
| 实测带宽 (GB/s) | 1571.8 | 1572.9 | 1573.5 | 1575.8 | 1579.9 | 819.7 |

对任一档，先计算各资源约束允许的 resident blocks 数，再取最小值：

\[
B_{\text{active}}=\min\left(
\left\lfloor\frac{T_{\text{SM,max}}}{256}\right\rfloor,
\left\lfloor\frac{S_{\text{SM}}}{S_{\text{block}}}\right\rfloor,
B_{\text{SM,max}},
B_{\text{register limit}},\ldots
\right),
\]

\[
\text{occupancy}=\frac{B_{\text{active}}\times256}{T_{\text{SM,max}}}.
\]

`S_block = 0` 时不能做 shared-memory 除法，该项视为不构成限制。还要考虑每 SM 最大 blocks、register 等约束，所以手算 shared/thread 两项后应以 occupancy API 的完整结果为准。

(a) 以 29.0 KB / block 这一档为例。程序输出表明本卡有 100 KB shared memory / SM，且最大常驻线程数为 1536：

\[
B_{\text{thread}}=\left\lfloor\frac{1536}{256}\right\rfloor=6,
\qquad
B_{\text{shared}}=\left\lfloor\frac{100}{29}\right\rfloor=3.
\]

在其他资源不构成更紧限制时，每 SM 可驻留 3 个 blocks，故

\[
\text{occupancy}=\frac{3\times256}{1536}=50\%.
\]

这与 occupancy API 输出的 3 blocks / SM、50.0% 一致。

(b) memory instruction 发出后，一个 warp 会等待较长的显存延迟；SM 可切换到其他 ready warps 继续发射。resident warps 减少不一定立即降低带宽，只要剩余 warps 仍足以覆盖延迟即可。本机在 3 blocks / SM（24 warps，50.0% occupancy）时仍维持约 1580 GB/s；降至 1 block / SM（8 warps，16.7%）后，可用于覆盖等待时间的独立请求不足，issue slots 更容易空闲，带宽才明显下降。

(c) 本卡的资源限制没有产生题面示例中的 75%、37.5% 和 12.5% 档位，因此应比较实际相邻档位。从 100.0%（15.0 KB 档，1573.5 GB/s）降至 83.3%（1575.8 GB/s），带宽反而增加约 0.15%，属于测量波动，说明高 occupancy 区域已经达到带宽平台。从 50.0% 的 1579.9 GB/s 降至 16.7% 的 819.7 GB/s，则减少 760.2 GB/s，即约 48.1%。这说明 occupancy 是隐藏延迟的手段而非性能的线性指标：超过所需并发度后继续增加 warps 收益很小，低于阈值后带宽才会快速下降。

## Module 5：计时与异步初步

### prob 5.1

(a) 报告纯 kernel 的设备执行耗时时，应使用 **`cudaEvent` 计时值**。events 记录在 CUDA stream 的设备时间线上，结束 event 只有在 kernel 完成后才会完成。

(b) 另外两项：

- host 计时、不等 GPU：主要测 host 发起/提交一次 kernel launch 的耗时，不是 kernel 执行时间。
- host 计时、等 GPU：测从 host 发起到 `cudaDeviceSynchronize` 返回的端到端等待时间，包含 kernel、launch/runtime 开销和同步开销，也可能包含设备排队等待。

三项具体数值为：

| 计时方法 | 耗时 (ms) |
| --- | ---: |
| host 计时、不等 GPU | 0.0061 |
| host 计时、等 GPU   | 0.0669 |
| cudaEvent 计时      | 0.0666 |

### prob 5.2

1. **对。** 同一 stream 内的操作按提交顺序执行。
2. **对。** 普通 kernel launch 相对 host 异步，提交后 host 通常继续执行；这不等于 launch 本身零开销。
3. **不能无条件判为对。** 在软件一致性的完整 Unified Memory 系统上，CPU 访问 GPU-resident managed page 通常会 page fault 并迁移；但 `directManagedMemAccessFromHost=1` 的硬件一致性系统可直接访问而不迁移。在 Windows/部分 Tegra 的受限模型中，GPU 活跃时 CPU 甚至不得访问 managed memory。即便平台支持并发访问，未同步地同时读写相同对象仍可能是数据竞争。

## Module 6：Tile 视角

### prob 6.1

1. **错。** array 才是 global memory 中所有 blocks 可见的对象；tile 是某个 tile block 本地的多维值/容器。通过 load 从 array 得到 tile，计算后再 store 回 array。
2. **对。** 程序员表达整个 tile 的操作，编译器把 tile 运算分配给 block 内硬件线程共同执行。
3. **错。** tile 与 SIMT kernels 可存在于同一程序中并分别启动；它们只是两种 kernel 编程模型。

### prob 6.2

|  | CUDA SIMT | cuTile | Triton |
| --- | --- | --- | --- |
| 并行单位 | block 里的 thread | tile block（程序员视角是一个逻辑线程） | program instance（通常对应一个 CTA/block） |
| 编号 | `blockIdx` / `threadIdx` | `ct.bid(axis)` | `tl.program_id(axis)` |
| 数据分工 | 用户用全局线程下标划分元素 | 用户按 `bid` 选择 tile；编译器把 tile 操作映射到 block 内线程 | 用户构造一组 offsets/tensor；编译器把它映射到线程 |
| 边界处理 | 用户写 `if` | array/view 知道 shape；用户选择 OOB load 的 padding policy，库/编译器实施屏蔽；store 丢弃 OOB | 用户显式构造 mask 并传给 `tl.load` / `tl.store` |

题给 cuTile 代码的 `tiled_view(...).load(...)` 没有指定 padding mode；当前文档中 Python 默认 OOB load 值是 undetermined，因此若向量长度不整除 `TILE`，严谨实现应明确选择 padding。不能仅凭这段代码说所有越界读取都自动补零。

### prob 6.3

(a) “每个硬件线程负责 tile 中哪些元素”由 cuTile 编译器和 built-in tile operations 决定；用户只选择 block 对应的 tile 及 tile shape。

(b) cuTile 代码中没有显式出现：`threadIdx`、`blockDim`、每线程全局元素下标、warp/lane、每 block 的线程数、显式的逐线程 load/store、合并访存安排、`__syncthreads()`。这些概念没有从硬件消失，只是由编译器/运行库负责 lowering。

## Module 7：TileLang 与 Triton

### prob 7.2：书面问答

改动主要集中在 kernel 的标量/逐元素计算表达式和新增标量参数。program id、offset tensor、mask、load/store 及 grid 都不需要变化，因为输入输出形状和“一次 program 处理一个 block of elements”的数据分工没有变化。Triton 对 tile tensor 做逐元素运算，融合多个逐元素算子不要求重新手写线程映射。

### prob 7.4：与 CUDA 2D kernel 对照

- 行、列仍有对应：现在是 tile 的全局起点 `by * block_M`、`bx * block_N`，而不是每个线程的行列号。
- grid 尺寸仍有对应：两个方向都要用 ceil-div 覆盖最后的 partial tile。
- 显式逐线程 `if (row < M && col < N)` 消失；本题指定的 `T.copy` 根据源/目标 Buffer 区域处理边缘搬运，相关安全访问由 lowering 完成。

### prob 7.5

| 谁负责 | CUDA SIMT | cuTile | Triton | TileLang |
| --- | --- | --- | --- | --- |
| 线程到数据的映射 | 用户 | 编译器 | 编译器；用户定义 program 的 offsets/shape | 用户定义并行迭代空间和线程数，编译器把迭代映射到线程 |
| 边界处理 | 用户显式判断 | 用户选 load padding policy，编译器执行；store 自动丢弃 OOB | 用户构造并传入 mask | `T.copy`/安全访问 pass 可自动处理可推导边界；其他不规则访问仍需用户 predicate |
| tile / block 尺寸选择 | 用户选择 block dimensions | 用户选择 tile shape，编译器决定内部线程数 | 用户选择 `BLOCK_SIZE` 等 meta-parameters；编译器负责 lowering | 用户选择 tile sizes、`threads`、pipeline stages 等 |
| block 内同步 | 用户 | 编译器/built-ins | 通常由编译器按 program 语义插入 | 普通 `T.copy` 场景通常由 `ThreadSync` pass 插入；手写 `T.async_copy` 时用户还要负责 wait，低层场景可显式同步 |

这里不能把任何 DSL 简化为“所有事情都归编译器”：Triton 的 mask 是明确的用户责任；TileLang 也暴露 threads、shared memory、pipeline 和低层异步控制。

### prob 7.6：为何 Triton 版未显式指定 shared/register/pipeline

Triton 程序表达 blocked tensors、`tl.load`、`tl.dot` 和 accumulator 的数据流，编译器根据 target、layout 与 meta-parameters 将它 lower 到寄存器、shared memory、tensor-core instructions 和调度。源码中的 `acc` 是逻辑 tensor value，不需要用户写 `alloc_fragment`；load 的复用和中间存储也由编译器分析。

“未显式写”不等于用户完全不影响它：`BLOCK_M/N/K`、`num_warps`、`num_stages` 等配置会显著改变寄存器压力、shared-memory 用量、pipeline 和 occupancy。TileLang 则把这些存储层次和流水结构更直接地暴露给用户。

### prob 7.8：TileLang 与 Triton softmax 的职责对照

- 归约：两边都是用户表达“求 max / sum”，编译器负责把 tile-level reduction lower 到 warp/block 指令；用户不手写完整归约树。
- 边界：Triton 用户显式生成 `offsets < N` mask，并为 max reduction 的 OOB lanes 选择合适的 neutral value；题给 TileLang contract 也要求用户把 fragment 补到 2 的幂并为无效位置填 `-inf`，安全存回可由 Buffer 边界 lowering 处理。
- 按形状编译：两者都做 JIT specialization。Triton 常由 `constexpr BLOCK_SIZE=next_power_of_2(N)` 形成特化；TileLang 题目要求 wrapper 按 `(M,N)` 生成并缓存特定 shape 的 kernel。形状策略和缓存 key 由用户组织，实际编译/lowering 由编译器完成。

#### Softmax 性能实测

测试设备为 NVIDIA GeForce RTX 5090，输入为 `(M,N)=(4096,N)` 的 FP32 tensor。首次 JIT 编译和正确性对拍在计时前完成；延迟由 `triton.testing.do_bench` 测量。有效带宽按最少一次输入读取和一次输出写入计算，即 `2 * M * N * sizeof(float) / time`，不代表 profiler 统计的实际 DRAM 流量。

| 行宽 N | Torch / ms | Triton / ms | TileLang / ms | Triton 相对 Torch | TileLang 相对 Torch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.0074 | 0.0081 | 0.0083 | 0.91x | 0.89x |
| 1024 | 0.0206 | 0.0230 | 0.0241 | 0.90x | 0.86x |
| 4096 | 0.0906 | 0.0889 | 0.0885 | 1.02x | 1.02x |

行宽 256 和 1024 时 Torch 分别比当前 Triton/TileLang 实现快约 10% 和 11%～16%；行宽 4096 时三者基本持平，单次实测中 Triton 和 TileLang 约快 2%。三版都将行内逐元素运算和归约融合在单个 kernel 中，主要受显存带宽和归约效率限制；行变宽后 launch 与调度开销更容易被摊薄。2% 的差距接近常见运行波动，不应在没有多轮统计或 profiler 数据时归因于某项具体编译优化。

## Module 8：平台与编译

### prob 8.1

1. **错。** PTX 是虚拟 ISA/高层汇编和中间表示，不是物理 GPU 直接执行的机器码；必须先编译为目标 SM 的 cubin/SASS。
2. **错。** cubin/SASS 只保证同一 compute capability 大版本内的有限二进制兼容；`sm_70` 与 CC 9.0 跨 major version，不兼容。
3. **对。** fatbin 可以同时包含多个 SM targets 的 cubins/SASS，以及一个或多个 PTX targets。
4. **对。** 应用运行时加载 PTX 时，由 NVIDIA device driver 完成 JIT，并通常缓存生成的 binary。

### prob 8.2

(a) 必须在实际 GPU 上记录，当前为 **待实测**。若可执行文件确实只含与本卡 major CC 不兼容的 `sm_90` SASS 且没有 PTX 兜底，预期 kernel launch 会返回“no kernel image is available for execution on the device”一类错误；这只是根据兼容规则给出的预期，不能冒充本机报错原文。

(b) 只含 `compute_75` PTX 的版本，在目标 GPU compute capability 不低于 7.5、且 driver 能理解该 PTX 版本等条件成立时可以运行。PTX 在应用加载模块/首次需要相应 device code 时由 driver JIT 为本机 SM 的机器码，结果通常进入 compute cache；首次运行会包含额外 JIT 成本。

### prob 8.3

Runtime API 是较高层的 `cuda*` 接口，替应用处理 primary context 等常规运行时细节，并与 CUDA C++ 语言和 kernel launch 语法紧密结合。Driver API 是较低层的 `cu*` 接口，显式管理 context、module、function、device memory 和 launch，适合需要动态装载与精细控制的框架/运行时。Runtime API 构建在 Driver API 之上；`cudaMalloc` 属于 Runtime API，对应低层 Driver API 家族中的 `cuMemAlloc` 等接口。

## Bonus：matmul 实验分析

所有性能数据均为 **待实测**。题面给出的 A100 数字只能作参考，不能作为本机结果。

- naive CUDA 每个输出线程独立从 global memory 读取一整行/列，没有用 shared tile 显式复用输入，算术强度和访存效率远低于 tiled 实现；它还是 FP32 FMA 路径，未利用本题 DSL 版本的 FP16 tensor cores。
- Triton/TileLang 通过二维 tiles 复用 A/B 数据、用 FP32 accumulator，并可 lower 到 tensor cores；tile shape 会同时影响数据复用、合并访存、tensor-core layout、register/shared-memory 压力、occupancy 和尾部浪费，所以必须实测调参。
- cuBLAS 有多架构专用 kernels、大量离线/运行时调优，以及成熟的异步搬运、流水和 tensor-core 使用，通常优于教学实现。
- TileLang 控制粒度更细只表示“允许用户表达更多优化”，不保证现有实现自动更快。当前仓库固定版本、候选配置、layout lowering、pipeline、同步、寄存器压力与编译器成熟度都可能让生成代码略逊于 Triton。只有用 profiler 比较实际指令、内存事务、stall reasons 和 occupancy 后，才能把某次差距归因到具体机制。

## 参考资料

以下均为官方资料，访问并核对于 2026-08-03：

1. [CUDA Programming Guide：Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)：block 独立性、warp/SIMT，以及 Thread Block Cluster 的调度范围。
2. [CUDA Programming Guide：Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#distributed-shared-memory)：线程层次、Distributed Shared Memory、同步、coalescing 和 occupancy。
3. [CUDA Programming Guide：Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#independent-thread-scheduling)：Volta/CC 7.0 后的 Independent Thread Scheduling。
4. [CUDA Programming Guide：Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)：streams、异步执行和同步。
5. [CUDA Programming Guide：Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)：不同平台的 managed-memory 迁移与一致性差异。
6. [CUDA Programming Guide：Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)：page fault、迁移和 host direct access 条件。
7. [CUDA Programming Guide：Writing Tile Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html)：cuTile 的 tile/block 映射和边界语义。
8. [CUDA Programming Guide：CUDA Platform](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)：PTX、cubin/fatbin、二进制兼容与 JIT。
9. [Triton 官方 Vector Addition 教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)：program id、offset tensor 和显式 mask。
10. [TileLang Language Basics](https://tilelang.com/programming_guides/language_basics.html)：`T.Kernel`、`T.Parallel`、`T.copy`、shared/fragment 与同步 lowering。
11. [CUDA Programming Guide：Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#sync)：`sync()` 的 barrier 与内存可见性保证，以及 cooperative grid launch。
