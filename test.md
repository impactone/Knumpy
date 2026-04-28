下面是优化后的完整版本。已按你的要求处理：

1. **设计思路中的用例全部从你给的 Benchmark 列表中选择**，删除或替换了原文中不在列表里的 `Choice`、`Shuffle`、`Permutation`、`Bounded`、`Median`、`Bincount` 等用例。
2. **RNG 与 Statistics 的功能边界更清晰**：RNG 聚焦 `bench_random` 与 `bench_xiecheng.RandomAndStats.time_rng`；Statistics 聚焦 `bench_core`、`bench_reduce`、`bench_function_base`、`bench_lib` 以及 `bench_xiecheng.RandomAndStats.time_statistics`。
3. **落地示意图保持原样**，只在图后增加更详细的原理解释。
4. **增加面向不熟悉 SE/性能优化背景人员的原理说明**，解释为什么这些路径能优化、为什么要回退、为什么不能随便改统计语义或随机数算法。

---

# 一、RNG 性能优化 SR — 功能实现设计

## 0 SR 描述、输入与输出

**SR 描述：**

本 SR 的功能是优化 NumPy 在鲲鹏 920B / Arm 平台上的随机数生成执行路径。它不新增随机数 API，不替换 `PCG64`、`MT19937`、`Philox`、`SFC64` 等随机数算法，也不改变固定 seed 下应有的算法语义；它要做的是在 NumPy 现有随机数链路中，对底层取数、整数范围映射、浮点归一化、正态分布快路径和连续输出写回进行平台化增强。

可以把这个 SR 理解为：

```text
用户仍然调用 NumPy 原有随机数接口
        |
        v
NumPy 保持原有随机数算法和分布语义
        |
        v
在适合的 Arm 场景下，底层生成和转换过程执行得更快
```

**输入：**

RNG SR 的输入不是业务数据数组，而是一次“随机数生成请求”。这个请求主要由以下信息组成：

| 输入类型 | 具体内容 | 说明 |
| --- | --- | --- |
| 随机源状态 | `Generator` / `RandomState` 内部 BitGenerator 状态，seed 初始化后的状态 | 例如 `PCG64`、`MT19937`、`Philox`、`SFC64`；优化不能改变状态推进顺序 |
| 随机接口参数 | `random`、`integers`、`randint`、`standard_normal` 等接口及其参数 | 决定走均匀浮点、整数、有界整数、正态分布等路径 |
| 输出规模 | `size`、shape、是否标量、输出元素数量 | 决定是否值得进入批量优化路径 |
| 输出类型 | `dtype`，例如 `uint32`、`uint64`、`float32`、`float64` | 决定使用哪类转换、写回和 dtype 专用路径 |
| 范围参数 | `low`、`high`、`endpoint`、整数范围大小 | 决定是否是满范围 fast path，还是需要拒绝采样 |
| 输出缓冲 | `out` 或 NumPy 内部分配的连续输出 buffer | 决定是否能连续写回，是否适合批量路径 |
| 平台能力 | Arm NEON/SVE 支持、CPU dispatch 结果、运行时开关 | 决定是否启用鲲鹏平台增强路径 |

对应到 benchmark，可以理解为：`RNG.time_32bit/time_64bit` 的输入重点是整数 dtype、范围和 size；`RNG.time_normal_zig` 的输入重点是 BitGenerator、size 和正态分布请求；综合用例 `time_rng(4000)` 需要进一步确认内部到底调用了哪类随机接口。

**输出：**

RNG SR 的功能输出仍然是 NumPy 原有随机数接口返回的结果，不新增对用户可见的新返回值。

| 输出类型 | 具体内容 | 约束 |
| --- | --- | --- |
| 随机数结果 | 标量或 `ndarray`，例如整数数组、`[0, 1)` 浮点数组、正态分布数组 | shape、dtype、取值范围、分布语义必须与 NumPy 原行为一致 |
| 随机源后续状态 | 生成请求完成后的 BitGenerator 内部状态 | 固定 seed 下的状态推进次数和输出序列语义不能被破坏 |
| 内部路径结果 | 是否命中 Arm 优化路径、是否回退原生路径 | 这是开发和验证信息，不改变 Python API |
| 性能结果 | benchmark 中体现为执行时间下降，perf 中体现为热点变化 | 性能收益必须由 benchmark 和 perf 证据支撑 |

因此，本 SR 的验收重点不是“输出一个新的随机数格式”，而是：**同样的输入请求，输出语义不变，但在鲲鹏 920B 上执行更快，并且能解释为什么更快**。

## 1 实现思路

本 SR 面向鲲鹏 920B 平台，对 NumPy 随机数生成链路中的热点路径进行性能优化。在**不改变上层 Python API、不改变随机数算法语义、不改变 seed 行为、不破坏结果正确性和随机数质量**的前提下，利用 Arm 平台可用的 NEON/SVE 向量能力、连续访存能力和平台感知分发能力，对随机 bit 生成、整数随机数生成、浮点随机数转换以及正态分布底层生成路径进行增强，从而提升随机数相关接口在鲲鹏平台上的整体执行效率。

本 SR 不改变 `PCG64`、`MT19937`、`Philox`、`SFC64` 等 BitGenerator 的算法定义，也不改变 NumPy 原有 `Generator` / legacy random 相关接口的语义。优化重点不是“换一个随机数算法”，而是在保证算法状态推进顺序、输出分布和统计性质不变的前提下，对底层批量执行过程进行平台化增强。

随机数生成链路可以理解为三层：

```text
上层接口：random / integers / normal 等
        |
        v
分布转换：整数范围映射、浮点归一化、normal 分布转换等
        |
        v
底层随机源：PCG64 / MT19937 / Philox / SFC64 / numpy legacy
```

本 SR 的优化重点主要落在第二层和第三层：即底层随机源生成得更快、生成后的整数/浮点转换更快，但不改变最终接口的功能语义。

面向“Arm 追平 Zen4”的设计不能只写成笼统的“向量化”。Zen4 和 Kunpeng920B 的乱序执行能力、分支预测、SIMD 宽度、整数乘法吞吐、cache 层次和编译器自动向量化效果都可能不同，因此本 SR 的方法分类应理解为**Arm 定向补强路径**：优先补 NumPy 源码中仍偏标量循环、逐元素分支、函数指针取数和小粒度写回的路径；已经由源码保证的语义能力，例如 `pcg64_next32` 复用 64-bit 输出的高低 32-bit，则不能再写成“新增算法”，而应写成“批量路径必须复用并保持该语义”。

为了方便开发理解，这里先把“随机数生成”拆成几个常见动作：

```text
1. 先由 BitGenerator 产生一串随机 bit
2. 如果用户要整数，就把随机 bit 映射到指定整数范围
3. 如果用户要 [0, 1) 浮点数，就截取有效 bit 并乘缩放系数
4. 如果用户要 normal 等分布，就继续套分布转换算法
```

不同 benchmark 测到的动作并不一样。例如 `time_32bit`、`time_64bit` 实际测的是整数生成，不是浮点转换；`time_normal_zig` 测的是正态分布生成；只有 `random()` / `random_sample()` 这类接口才主要测 `[0, 1)` 浮点生成。开发时需要先确认用例落在哪个动作上，再决定优化函数入口、分布转换逻辑，还是底层 BitGenerator。

---

### 1.1 BitGenerator 随机源供给路径优化

这一类功能主要负责为上层随机数接口提供底层随机 bit 输出能力，是 `random`、`integers`、`normal` 等接口共享的基础供给路径。可以把它理解成随机数链路里的“发动机”：上层要整数、浮点数还是正态分布，最开始都要先从这里拿随机 bit。

随机 bit 可以理解为随机数生成链路中的“原材料”。例如：

```text
生成随机 bit
   |
   +--> 转成 uint32 / uint64
   |
   +--> 转成 [0, 1) 浮点数
   |
   +--> 进一步变换成 normal 等分布
```

因此，只要底层随机 bit 的批量供给效率提升，上层多个接口都有机会获得收益。

这里最重要的约束是：随机数不是随便并行多算几个就可以。`PCG64`、`MT19937`、`SFC64` 等 BitGenerator 都有自己的内部状态，生成第 `i+1` 个随机数通常依赖前一个状态。如果优化改变了状态推进顺序，即使结果看起来仍然“随机”，也会破坏固定 seed 下的输出序列。因此这一类优化必须在**不改变算法、不改变 seed 行为、不改变状态推进顺序**的前提下进行。

从开发角度看，优化空间主要有两类：

```text
函数层优化：减少上层调用和分发开销
底层优化：减少循环开销、提高连续写回效率、优化状态更新和取数路径
```

对于大数组随机数生成，主要耗时一般在底层取数和写回循环里，函数层优化只能带来较小收益。对于小数组或频繁调用场景，函数入口的参数解析、dtype 判断和分发成本才会更明显。

需要特别强调的是，本优化不是把随机算法改成另一个算法，也不是改变随机数质量，而是让同样的算法在大批量输出时执行得更高效。

**相关用例：**

* `bench_random.RNG.time_raw('MT19937')`
* `bench_random.RNG.time_raw('PCG64')`
* `bench_random.RNG.time_raw('Philox')`
* `bench_random.RNG.time_raw('SFC64')`
* `bench_random.RNG.time_raw('numpy')`
* `bench_random.RNG.time_32bit('MT19937')`
* `bench_random.RNG.time_32bit('PCG64')`
* `bench_random.RNG.time_32bit('Philox')`
* `bench_random.RNG.time_32bit('SFC64')`
* `bench_random.RNG.time_32bit('numpy')`
* `bench_random.RNG.time_64bit('MT19937')`
* `bench_random.RNG.time_64bit('PCG64')`
* `bench_random.RNG.time_64bit('Philox')`
* `bench_random.RNG.time_64bit('SFC64')`
* `bench_random.RNG.time_64bit('numpy')`

---

### 1.2 批量整数随机数生成路径优化

这一类功能主要负责生成整数随机数，包括常见的 32-bit、64-bit 随机整数，以及不同 BitGenerator 下的整数输出能力。整数随机数生成既可以直接服务上层整数采样，也可以作为其他随机分布转换的中间输入。

这类 benchmark 实际测的是 `np.random.randint`、`Generator.integers`、legacy `RandomState.randint` / `random_integers` 这些整数接口。其中 `time_raw` 虽然名字里有 `raw`，但在当前 benchmark 中实际也是调用整数生成接口，并不是直接导出底层 raw bit。

从原理上看，有界整数采样通常不能简单地对随机 bit 取模，因为取模可能引入分布偏差。例如生成 `[0, 10)` 的整数，如果直接对随机数 `% 10`，某些结果可能出现概率略高。为了保持均匀分布，常见实现会使用拒绝采样：先生成候选值，若候选值落在会导致偏差的区间内，则丢弃并重新生成。这个过程如果逐元素执行，会有较多分支和循环补齐开销。因此，本 SR 通过批量生成候选值、批量判断合法性、批量写回合法结果来降低开销。

可以用一个更直观的例子理解“为什么不能简单取模”：

```text
假设随机源只能均匀产生 0..15，一共 16 个数
现在想得到 0..9，一共 10 个数
如果直接 value % 10：
0..5 会各出现 2 次
6..9 只会各出现 1 次
结果就不均匀
```

因此整数随机数生成通常分两种情况看：

```text
满范围整数：例如 uint32 的 0..2^32-1，基本可以直接取底层随机 bit
有界整数：例如 0..10、0..2^30+1，需要范围映射和拒绝采样
```

满范围整数更接近规则循环，优化重点是减少循环和写回开销；有界整数的难点在于候选值可能被拒绝，每个输出消耗的随机 bit 数量不固定，所以更适合做保守的批量化，而不是简单假设可以完全 SIMD 化。

**相关用例：**

* `bench_random.Randint.time_randint_fast`
* `bench_random.Randint.time_randint_slow`
* `bench_random.Randint_dtype.time_randint_fast('uint64')`
* `bench_random.Randint_dtype.time_randint_slow('uint64')`
* `bench_random.RNG.time_32bit('MT19937')`
* `bench_random.RNG.time_32bit('PCG64')`
* `bench_random.RNG.time_32bit('Philox')`
* `bench_random.RNG.time_32bit('SFC64')`
* `bench_random.RNG.time_64bit('MT19937')`
* `bench_random.RNG.time_64bit('PCG64')`
* `bench_random.RNG.time_64bit('Philox')`
* `bench_random.RNG.time_64bit('SFC64')`

---

### 1.3 随机浮点转换路径优化

这一类功能主要负责把底层生成的随机整数或随机 bit 转换成上层可直接使用的随机浮点数，是均匀分布随机数生成的关键基础能力。其本质功能是完成从整数随机源到浮点随机数的映射和归一化。

例如生成 `[0, 1)` 区间的 `float64` 随机数时，底层通常先产生一定数量的随机 bit，然后通过位移、掩码、缩放因子等方式转换为浮点数。这个过程可以抽象为：

```text
随机 bit / 随机整数
        |
        v
截取有效随机位
        |
        v
乘以归一化系数
        |
        v
得到 [0, 1) 浮点随机数
```

如果生成的是大数组，这一转换会重复执行很多次，因此非常适合做批量化处理。

需要注意的是，这一类优化对应的是 `random()` / `random_sample()` 这类 `[0, 1)` 均匀浮点接口，而不是 `time_32bit`、`time_64bit` 这类整数 benchmark。`time_32bit`、`time_64bit` 生成的是 `uint32` / `uint64`，应该归到整数随机数路径。

浮点随机数转换相对容易理解：底层先生成一个足够长的随机整数，然后取其中高质量的若干 bit，乘以一个固定比例，把它压到 `[0, 1)` 区间。例如 `float64` 只需要有限个随机有效位，不会把 64 个 bit 全部直接塞进小数。这个过程规则、分支少、输出连续，因此比有界整数和正态分布更适合作为向量化或批量化目标。

本 SR 的思路是将位移、掩码、归一化和连续写回等步骤做批量化处理，减少逐元素标量转换和重复访存。函数层可以优化 dtype 和 out 参数的分发，核心收益仍然主要来自底层 fill 循环。

**相关用例：**

* `bench_xiecheng.RandomAndStats.time_rng(4000)`

说明：当前 `bench_random.RNG.time_32bit` / `time_64bit` 生成的是整数，不属于本小节的浮点转换路径。如果 `bench_xiecheng.RandomAndStats.time_rng(4000)` 内部包含 `random()` / `random_sample()` / `uniform()` 这类调用，则可作为本路径验证用例；否则应新增或改用明确的均匀浮点随机数 benchmark。

---

### 1.4 正态分布采样底层路径优化

这一类功能主要负责优化正态分布随机数生成中的热点路径。正态分布是统计、仿真、机器学习场景中的高频分布之一。NumPy 中正态分布生成通常会依赖底层均匀随机源，再通过特定算法转换为正态分布样本。

本 SR 的优化重点不是替换正态分布算法，而是在保证原有算法语义、分布性质和边界行为不变的前提下，增强底层随机源供给以及正态分布转换中可安全批量化的局部步骤。

这里需要特别说明：正态分布不是简单地“随机生成一个数”即可，它需要满足特定的概率密度分布。如果随意改动转换公式，可能导致均值、方差、尾部概率等统计性质发生变化。因此该路径的优化必须保守，优先做底层供给增强和局部批量化，而不是重写分布算法。

NumPy 的 `standard_normal` 走的是 Ziggurat 类算法。可以把它理解为：

```text
大多数样本：随机 bit 命中快速区域，直接算出结果
少数样本：落到边界或尾部区域，需要额外判断、重试、log/exp 等计算
```

这也是它不如均匀浮点数那样容易向量化的原因：大多数情况很快，但少数情况会走复杂分支，而且不同元素可能重试次数不同。开发时可以优先优化快路径、外层 fill 循环和底层随机源供给；尾部和拒绝采样分支要保持原生语义或谨慎回退。

**相关用例：**

* `bench_random.RNG.time_normal_zig('MT19937')`
* `bench_random.RNG.time_normal_zig('PCG64')`
* `bench_random.RNG.time_normal_zig('Philox')`
* `bench_random.RNG.time_normal_zig('SFC64')`
* `bench_random.RNG.time_normal_zig('numpy')`
* `bench_xiecheng.RandomAndStats.time_rng(4000)`

---

### 1.5 RNG 与下游统计链路联动验证

这一类功能主要用于验证优化后的随机数结果进入下游统计分析场景后，是否仍然保持正确、稳定和一致。它不是直接生成随机数的功能，而是面向验收和链路评估的支撑部分。

随机数在业务下游往往不会单独存在，而是继续参与均值、方差、标准差、最大值、分位数、直方图等统计分析。例如仿真或机器学习场景中，随机数生成之后通常马上做统计汇总。因此，RNG 优化除了看生成速度，还需要看随机数据进入统计链路后是否表现稳定。

这一类设计的重点不在于新增 RNG 算法能力，而在于确保性能收益建立在**结果正确、随机性质量稳定、下游行为一致**的基础之上。

**相关用例：**

* `bench_xiecheng.RandomAndStats.time_rng(4000)`
* `bench_xiecheng.RandomAndStats.time_statistics(4000)`
* `bench_reduce.StatsReductions.time_mean('float32')`
* `bench_reduce.StatsReductions.time_mean('float64')`
* `bench_reduce.StatsReductions.time_std('float32')`
* `bench_reduce.StatsReductions.time_std('float64')`
* `bench_function_base.Histogram1D.time_full_coverage`
* `bench_function_base.Histogram1D.time_small_coverage`
* `bench_function_base.Histogram1D.time_fine_binning`
* `bench_function_base.Histogram2D.time_full_coverage`
* `bench_function_base.Histogram2D.time_small_coverage`
* `bench_function_base.Histogram2D.time_fine_binning`
* `bench_function_base.Percentile.time_percentile`
* `bench_function_base.Percentile.time_percentile_small`
* `bench_function_base.Percentile.time_quartile`

---

### 1.6 RNG 主要优化方法与原理

RNG SR 的核心不是“把随机数算法换成向量算法”，而是在保持随机数算法、seed 序列和分布语义不变的前提下，优化随机数从底层 bit 到最终结果的执行方式。

这里需要先校正分类边界：下面这些不是彼此完全独立的源码模块，而是一条随机数生成链路上的不同优化方法。`BitGenerator` 供给是底层公共能力，`integers`、`random`、`standard_normal` 会在它之上继续做整数映射、浮点归一化或分布转换。因此某个 benchmark 可能同时受多个方法影响，但它最终测的 API 仍要按源码入口判断。

源码依据如下：

| 源码入口 | 对应功能 | 支撑的优化方法 |
| --- | --- | --- |
| `numpy/random/_generator.pyx::Generator.integers` | 整数随机数入口 | 整数满范围 fast path、有界整数拒绝采样批量化 |
| `numpy/random/src/distributions/distributions.c::random_bounded_uint64_fill` | 有界整数批量填充 | 有界整数拒绝采样批量化 |
| `numpy/random/_generator.pyx::Generator.random` | `[0, 1)` 均匀浮点入口 | 均匀浮点转换批量化 |
| `numpy/random/_generator.pyx::Generator.standard_normal` | 正态分布入口 | Ziggurat 快路径增强 |
| `numpy/random/src/distributions/distributions.c::random_standard_normal` | Ziggurat 正态分布核心 | Ziggurat 快路径增强、慢路径保守回退 |
| `numpy/random/src/distributions/distributions.c::random_standard_uniform_fill` | 均匀浮点数组填充，源码是 `for` 循环逐个调用 `next_double` | fill 循环批量化、连续写回 |
| `numpy/random/src/distributions/distributions.c::random_standard_normal_fill` | 正态分布数组填充，源码是 `for` 循环逐个调用 `random_standard_normal` | 外层 fill 循环优化、Ziggurat 快路径组织 |
| `numpy/random/_pcg64.pyx::PCG64.__init__` | 绑定 `next_uint64/next_uint32/next_double` | BitGenerator 批量供给优化 |
| `numpy/random/src/pcg64/pcg64.h::pcg_setseq_128_step_r` | PCG64 状态推进：128-bit LCG 乘加 | BitGenerator 状态更新优化，但不能改变推进次数 |
| `numpy/random/src/pcg64/pcg64.h::pcg_output_xsl_rr_128_64` | PCG64 输出混洗：`high ^ low` 后按高位旋转 | 输出 permutation 执行优化，但不能改变 bit 序列 |
| `numpy/random/src/pcg64/pcg64.h::pcg64_next32` | 复用一次 64-bit 输出的两个 32-bit 半区 | 32-bit 整数路径保持源码已有缓存语义 |

#### 面向 Arm 追平 Zen4 的 RNG 优化重点

现有分类方向基本正确，但还不够“Arm 定向”。需要把每类方法写清楚：它在源码里卡在哪里、Arm 上可能输在哪里、开发应该补哪一刀，以及哪些地方不能承诺优化。

| 优化方法 | Arm 追平 Zen4 的主要抓手 | 源码依据与边界 |
| --- | --- | --- |
| BitGenerator 批量供给优化 | 减少函数指针取数、循环控制、状态结构反复读写和小粒度 store；用分块展开提高 Kunpeng 上的指令流水与连续写回效率 | `bitgen_t` 通过 `next_uint64/next_uint32/next_double` 函数指针供给随机源；PCG64 状态有前后依赖，不能直接把一个状态流拆成任意 SIMD 多路并行 |
| 整数满范围 fast path | 满范围整数尽量走直接取 bit + 连续写回；`uint32` 路径必须利用已有 64-bit 拆两次 32-bit 的缓存语义，避免多取随机源 | `pcg64_next32` 已有 `has_uint32/uinteger` 缓存；开发重点是批量 fill 路径不要绕开该语义，不是重新设计 32-bit 随机算法 |
| 有界整数拒绝采样批量化 | 降低逐元素 Lemire/masked 拒绝采样的分支成本；候选生成、合法性判断、写回按块组织，减少 Arm 上分支不稳定带来的流水线损失 | `random_bounded_uint64_fill` 中按 `rng` 大小选择直接路径、32-bit Lemire、masked 或 64-bit 路径；拒绝采样规则不能简化成 `% range` |
| 均匀浮点转换批量化 | 对 `next_uint64 -> 取有效位 -> 转浮点 -> 乘缩放系数 -> store` 这条规则路径做 NEON/SVE 友好的批量转换和连续写回 | `random_standard_uniform_fill` 现在是逐元素 fill；`float64` 有 53 个有效随机位，`float32` 有 24 个有效随机位，不能混用 |
| Ziggurat 快路径增强 | 优先优化 99% 左右命中的快路径：bit 拆分、表查找、符号处理、连续写回；慢路径的 `log/exp` 和重试逻辑保守回退 | `random_standard_normal` 快路径 `rabs < ki_double[idx]` 直接返回，尾部路径会额外消耗随机数；不能为了对齐 SIMD 改变每个样本消耗随机数的顺序 |

因此，这里不需要把五类优化路径推倒重写，但需要把“向量化”收窄成几种可开发的方法：**分块循环展开、平台 SIMD 转换、连续写回、减少分支、函数入口 fast path、严格回退**。尤其是 PCG64 这类有状态 BitGenerator，不适合承诺“完全 SIMD 化随机源本身”；更合理的目标是让状态推进仍按原顺序执行，同时减少外层 fill、转换和写回开销。`Philox` 这类 counter-based 随机源理论上更容易并行，但本 SR 仍不能通过更换默认随机源或改变输出顺序来获得性能。

主要方法如下。

#### 方法一：BitGenerator 批量供给优化

该方法面向 `PCG64`、`MT19937`、`Philox`、`SFC64` 等底层随机源。BitGenerator 的作用是产生原始随机 bit，例如 `next_uint64`、`next_uint32`、`next_double`。上层的 `integers`、`random`、`standard_normal` 都会消耗这些底层随机 bit。

原理上，BitGenerator 每生成一个值都要完成状态读取、状态推进、输出混洗和结果返回。以 `PCG64` 为例，源码里的核心链路可以简化理解为两步：

```text
状态推进：
    state = state * multiplier + inc

输出混洗：
    output = rotate_right(state.high ^ state.low, state.high >> 58)
```

“状态推进简单稳定 + 输出时做 permutation 混洗”的意思是：内部状态按一个固定公式往前走，像钟表齿轮一样每次只走确定的一格；但直接把内部状态暴露出来随机性不够好，所以输出前再把高低位异或、旋转，把 bit 打散后交给用户。优化时可以让这套齿轮转得更省循环、更少函数开销，但不能跳齿、不能换齿轮，也不能改变混洗方式，否则固定 seed 下的输出序列就会变化。

优化方法不是改变状态公式，而是优化执行组织方式：

```text
原始方式：生成一个 -> 写回一个 -> 再生成下一个
优化方式：按块生成 -> 局部暂存 -> 连续写回 -> 尾块标量收尾
```

这样做能减少循环控制、函数调用、状态结构反复访问和小粒度写回开销。该方法对 `RNG.time_raw`、`RNG.time_32bit`、`RNG.time_64bit` 都有间接影响，但这些 benchmark 的入口仍是整数生成接口，不能把它们简单等同于纯 raw bit benchmark。

#### 方法二：整数随机数满范围 fast path

该方法面向 `uint32`、`uint64` 满范围或近满范围整数输出，例如 `RNG.time_32bit`、`RNG.time_64bit`。这类用例本质上是把底层随机 bit 直接写成目标整数 dtype，范围映射较少，执行路径更规则。

原理上，如果目标范围刚好覆盖 dtype 的完整范围，例如 `uint32` 的 `0..2^32-1`，就不需要复杂的拒绝采样。底层生成的随机 bit 可以直接作为结果，或者把一个 64-bit 输出拆成两个 32-bit 结果使用。

优化方法包括：

```text
1. 对满范围 uint32 / uint64 建立直接输出路径
2. 保持源码已有的 64-bit 输出拆两个 32-bit 半区语义，减少底层取数次数
3. 按 dtype 连续写回，减少逐元素分发和类型转换
4. 对小数组保持原生路径，避免 fast path 准备成本超过收益
```

这个方法的收益来自“少做多余判断”。它适合规则整数输出，不适合任意上下界的有界整数采样。

#### 方法三：有界整数拒绝采样批量化

该方法面向 `Randint.time_randint_fast`、`Randint.time_randint_slow`、`Randint_dtype.time_randint_*` 以及部分 `integers(low, high)` 场景。它解决的是“随机整数必须均匀落在指定范围内”的问题。

有界整数不能直接用 `% range`，因为随机源范围和目标范围通常不能整除，会导致某些结果出现概率更高。NumPy 需要通过 mask、Lemire 方法或拒绝采样避免分布偏差。

优化方法是把“一个一个生成、一个一个判断”改成“批量生成候选值、批量判断、批量写回合法值”：

```text
生成一批候选随机整数
        |
        v
批量判断是否落在无偏范围内
        |
        v
合法结果连续写回
        |
        v
不足的部分继续批量补齐
```

该方法的原理是减少分支和循环补齐成本，但不改变拒绝采样规则。候选值被拒绝时，仍然必须继续生成补齐，不能为了性能跳过拒绝采样，否则会改变整数分布。

#### 方法四：均匀浮点转换批量化

该方法面向 `random()`、`random_sample()`、`uniform()` 这类 `[0, 1)` 均匀浮点生成。当前公开 `bench_random.RNG.time_32bit` / `time_64bit` 是整数用例，不能直接证明该方法收益；如果 `bench_xiecheng.RandomAndStats.time_rng(4000)` 内部包含均匀浮点随机数生成，则可以作为验证用例。

浮点随机数不是直接从硬件生成一个 `float64`。通常做法是先生成随机整数，再取其中一部分有效 bit，乘以固定缩放因子，把结果映射到 `[0, 1)`。例如 `float64` 需要 53 个有效随机位。

优化方法包括：

```text
1. 批量获取随机 bit
2. 批量右移，截取 float32 / float64 所需有效位
3. 批量乘以归一化系数
4. 连续写回 float 输出 buffer
```

该方法分支少、规则性强，适合批量化。边界语义必须保持一致，例如包含 0、不包含 1，`float32` 和 `float64` 的有效位数量不能混用。

#### 方法五：Ziggurat 正态分布快路径增强

该方法面向 `RNG.time_normal_zig`，实际测的是 `standard_normal`。NumPy 使用 Ziggurat 类算法生成正态分布。这个算法大多数样本会落在快速区域，少数样本会进入边界或尾部处理。

原理可以理解为：

```text
大多数样本：
    随机 bit 选中一个矩形区域，快速判断通过，直接返回结果

少数样本：
    落在边界或尾部区域，需要额外随机数、log/exp、重试判断
```

优化方法应集中在快路径：

```text
1. 优化快路径中的随机 bit 获取和表查找
2. 优化符号位、索引、有效位拆分
3. 优化外层 fill 循环和连续写回
4. 慢路径保持原生算法，必要时直接回退
```

该方法不能替换正态分布算法，也不能改变尾部概率。正态分布的均值、方差和尾部行为都依赖算法细节，随意改公式会破坏随机数质量。

#### RNG 方法与用例对应关系

| 主要方法 | 覆盖用例 | 主要收益来源 |
| --- | --- | --- |
| BitGenerator 批量供给优化 | 作为 `RNG.time_raw`、`RNG.time_32bit`、`RNG.time_64bit`、`RNG.time_normal_zig` 的底层支撑 | 减少底层取数、状态访问、循环和写回成本 |
| 整数满范围 fast path | `RNG.time_32bit`、`RNG.time_64bit` | 减少范围判断和 dtype 写回开销 |
| 有界整数拒绝采样批量化 | `Randint.*`、`Randint_dtype.*`、部分 `RNG.time_raw` | 减少逐元素拒绝采样分支和补齐成本 |
| 均匀浮点转换批量化 | `random` / `random_sample` / `uniform` 相关综合用例；当前 `time_32bit/time_64bit` 不能证明该路径收益 | 批量完成 bit 截取、缩放和写回 |
| Ziggurat 快路径增强 | `RNG.time_normal_zig` | 优化高概率快路径，慢路径保持语义 |

---

## 2 实现设计

基于上述思路，本 SR 的实现按具体优化类型分别落地。实现设计围绕“具体功能、落地方法、原理解释、优化前后效果对比”进行描述，确保开发、测试和 SE 都能理解优化目标、启用边界和风险控制方式。

---

### 2.1 BitGenerator 随机源供给优化实现

该部分实现的功能，是在保持原有随机数算法定义不变的前提下，提高底层随机 bit 的批量输出效率，为上层各类随机数接口提供更高吞吐的基础随机源。

在 `PCG64`、`MT19937`、`Philox`、`SFC64` 等热点 BitGenerator 上，保持 NumPy 原有算法定义、seed 初始化方式和状态推进顺序不变，仅对底层取数过程进行增强。具体做法是在底层 C 实现中增加面向鲲鹏平台的优化路径，对随机源供给采用**分块循环展开、批量暂存和连续内存写回**的方式处理；当请求长度达到阈值且当前场景适合优化时，优先进入优化路径，否则根据条件选择原生路径。

实现中要求尾块处理保持与原生行为一致，即当最后一段数据不足一个批量处理宽度时，使用标量路径收尾，避免因边界处理改变输出行为。该实现主要提升底层随机源吞吐，并直接支撑上层各类随机接口。

**落地示意：**

```text
优化前：逐元素或小批量生成、逐段写回

请求 N 个随机 bit
        |
        v
+----------------------+
| 逐元素/小批量生成     |
+----------------------+
        |
        v
+----------------------+
| 逐元素/小批量写回     |
+----------------------+
        |
        v
输出 buffer


优化后：分块生成、批量暂存、连续写回

请求 N 个随机 bit
        |
        v
+------------------------------+
| 按固定块大小切分 Block 0..k  |
+------------------------------+
        |
        v
+------------------------------+
| 每个 Block 内循环展开生成    |
| 一次处理多个随机 bit         |
+------------------------------+
        |
        v
+------------------------------+
| 批量暂存在寄存器/局部缓冲中 |
+------------------------------+
        |
        v
+------------------------------+
| 连续写回到输出 buffer        |
+------------------------------+
        |
        v
+------------------------------+
| 尾块不足批量宽度时标量收尾  |
+------------------------------+
```

**原理解释：**

随机 bit 生成通常会重复执行同一套状态更新和输出逻辑。如果每次只生成一个值，就会反复执行循环判断、函数调用、状态读写和结果写回。对于大数组来说，这些“控制成本”会被重复放大。

分块处理的思想是：一次处理一批数据，而不是每次只处理一个数据。块内通过循环展开减少循环次数，通过批量暂存减少中间写回，通过连续写回提升 cache 友好性。这样做不会改变随机数算法本身，只是改变执行组织方式。

尾块必须使用标量逻辑收尾，是因为最后剩余元素数量可能不足一个批量宽度。如果强行按批量宽度写回，可能越界或生成多余结果；如果改变状态推进次数，可能导致后续随机数序列与原生实现不一致。因此尾块处理是保证正确性的关键点。

需要注意，当前 `RNG.time_raw`、`RNG.time_32bit`、`RNG.time_64bit` 虽然都会消耗底层随机源，但 benchmark 入口仍然是整数生成接口。因此这些用例能反映底层随机源供给能力，也会同时受到整数范围映射、dtype 处理和 legacy / Generator 分发差异影响。开发分析 perf 时要把“底层取数热点”和“整数采样热点”分开看。

**优化前后效果对比：**

| 对比项     | 优化前                                                      | 优化后                             |
| ------- | -------------------------------------------------------- | ------------------------------- |
| 处理方式    | 以原生通用路径为主，随机 bit 输出偏逐元素或小批量处理                            | 按块组织生成过程，采用批量生成、批量暂存和连续输出       |
| 循环/分支开销 | 大批量生成时循环次数多，循环控制和状态推进开销较明显                               | 通过分块循环展开减少循环控制成本，降低重复调度开销       |
| 访存/写回模式 | 输出 buffer 虽然连续，但写回粒度较小，对平台能力利用不足                         | 批量暂存后连续写回，提高写回效率和缓存友好性          |
| 适用场景    | 小规模请求和通用随机 bit 输出场景                                      | 大批量、连续输出、规则 buffer 的随机 bit 生成场景 |
| 预期效果    | 底层随机源供给能力会影响 `integers`、`random`、`normal` 等上层接口 | 底层取数吞吐提升，并带动上层随机接口受益       |

---

### 2.2 批量整数随机数生成优化实现

该部分实现的功能，是在指定 dtype 和范围约束下，提高整数随机数生成效率，支撑整数随机数相关场景。

具体做法包括：对 `uint64` 以及 32-bit / 64-bit 原始输出路径建立批量生成 fast path，对范围映射中的常见分支进行预判，在有界整数场景下用批量判断替代逐元素判断，并对拒绝采样结果采用“批量补齐 + 局部收尾”的方式控制额外开销。

实现上对 fast path 与 slow path 分别处理：对于易于直接映射的路径，采用连续批量生成与一次性写回；对于存在拒绝采样的路径，使用批量比较与筛选保留合法结果，并在不足时增量补充，直到填满目标缓冲区。对极小数组、复杂 dtype 或当前平台不满足优化条件的情况，则直接走原生路径。

**落地示意：**

```text
优化前：逐元素生成和判断

目标输出 M 个整数
        |
        v
+----------------------+
| 生成一个随机整数      |
+----------------------+
        |
        v
+----------------------+
| 判断是否落入范围      |
+----------------------+
        |
        +---- 不合法 ----> 重新生成
        |
       合法
        |
        v
+----------------------+
| 写入输出 buffer       |
+----------------------+
        |
        v
重复直到填满 M 个整数


优化后：批量生成、批量筛选、批量补齐

目标输出 M 个整数
        |
        v
+------------------------------+
| 一次生成一批候选随机整数     |
+------------------------------+
        |
        v
+------------------------------+
| 批量完成范围判断和筛选       |
+------------------------------+
        |
        v
+------------------------------+
| 合法结果批量写入输出 buffer |
+------------------------------+
        |
        v
+------------------------------+
| 若未填满，则继续批量补齐     |
+------------------------------+
```

**原理解释：**

整数随机数生成看起来只是“生成一个整数”，但如果要求结果落在某个范围内，就必须处理分布均匀性问题。直接取模虽然简单，但在很多范围下会引入偏差。为了保证每个整数出现概率一致，NumPy 这类库通常会使用拒绝采样。

拒绝采样的特点是：有些候选值会被丢弃，需要重新生成。逐元素处理时，每个元素都要单独判断是否合法，如果不合法还要单独补齐，分支开销较高。批量处理后，可以一次生成多个候选值，再统一判断哪些合法，合法结果批量写入，不足部分再批量补齐。这样既保持了均匀性，又减少了逐元素分支判断和循环次数。

需要注意的是，批量补齐不能改变最终输出数量，也不能改变合法值的概率分布。因此实现中必须保证筛选规则与原生逻辑一致，不能为了性能直接跳过拒绝采样。

**优化前后效果对比：**

| 对比项     | 优化前                                           | 优化后                          |
| ------- | --------------------------------------------- | ---------------------------- |
| 处理方式    | 整数随机数生成偏逐元素生成、逐元素范围映射                         | 对常见 dtype 建立批量生成路径，批量完成范围映射  |
| 循环/分支开销 | 有界整数和拒绝采样中分支判断较多，补齐过程重复开销明显                   | 使用批量比较、批量筛选和批量补齐，减少分支和循环控制成本 |
| 访存/写回模式 | 合法结果逐步写入，写回粒度较小                               | 合法结果集中筛选后批量写回，提高输出效率         |
| 适用场景    | 通用整数采样、小规模请求、复杂 dtype 场景                      | 大批量整数采样、常见 dtype、有界整数采样场景    |
| 预期效果    | `Randint.*`、`Randint_dtype.*` 相关路径在大批量场景下开销较高 | 大批量整数随机数生成吞吐提升，有界整数场景收益更明显   |

---

### 2.3 随机浮点转换优化实现

该部分实现的功能，是将底层整数随机源高效转换为浮点随机数结果，支撑均匀随机数相关路径。

具体做法是，在底层生成整数随机 bit 之后，对位移、掩码提取、归一化因子计算和区间映射等步骤进行批量化处理，尽量减少中间临时变量和重复访存；在连续数组场景中采用顺序加载与顺序写回，提高整体缓存利用率和执行效率。

该实现主要对应 `random()`、`random_sample()`、`uniform()` 等均匀浮点接口。如果当前 benchmark 集合中没有直接覆盖这些接口，需要通过内部综合用例确认是否真的调用了均匀浮点生成；不能用 `time_32bit`、`time_64bit` 的整数结果直接证明浮点转换路径收益。

这一实现要求与原生路径保持数值语义一致，尤其对区间定义、dtype 精度和边界值处理必须严格对齐。尾块部分继续使用标量逻辑收尾，确保行为一致。

**落地示意：**

```text
优化前：整数随机源逐步转换为浮点随机数

随机 bit / 随机整数
        |
        v
+------------------+
| 位移 / 掩码处理   |
+------------------+
        |
        v
+------------------+
| 标量归一化转换   |
+------------------+
        |
        v
+------------------+
| 区间映射          |
+------------------+
        |
        v
写入 float 输出


优化后：批量完成转换链路

一批随机 bit / 随机整数
        |
        v
+----------------------------+
| 批量位移 / 批量掩码处理    |
+----------------------------+
        |
        v
+----------------------------+
| 批量归一化为 float32/64    |
+----------------------------+
        |
        v
+----------------------------+
| 批量完成区间映射           |
+----------------------------+
        |
        v
+----------------------------+
| 连续写回 float 输出 buffer |
+----------------------------+
```

**原理解释：**

浮点随机数通常不是直接从硬件“生成一个 float”，而是先生成随机 bit，再将其中一部分 bit 映射到浮点数有效位上，最后乘以一个缩放系数，使结果落在 `[0, 1)` 或目标区间内。

这个转换链路有几个关键约束：

* 生成的浮点数不能包含目标区间外的值。
* `float32` 和 `float64` 的有效位数量不同，转换规则不能混用。
* 边界值处理必须与 NumPy 原生实现一致，例如是否包含 0、是否排除 1。
* 批量化不能改变随机 bit 的消费顺序，否则同一个 seed 下的结果序列可能变化。

因此，本优化只改变执行方式：把多个元素的位移、掩码、归一化和写回集中处理，而不改变每个元素的数学定义。

**优化前后效果对比：**

| 对比项     | 优化前                         | 优化后                                  |
| ------- | --------------------------- | ------------------------------------ |
| 处理方式    | 整数到浮点转换主要依赖通用转换路径，逐元素处理成本较高 | 对位移、掩码、归一化和区间映射进行批量化处理               |
| 循环/分支开销 | 高频调用时转换循环重复执行，标量处理占比高       | 批量完成转换步骤，减少重复循环和标量指令占比               |
| 访存/写回模式 | 中间步骤较多，可能产生额外临时变量和重复访存      | 减少中间变量，采用顺序加载和顺序写回                   |
| 适用场景    | 通用浮点随机数生成、小规模请求             | 大规模 `float32/float64` 随机数生成、连续数组输出场景 |
| 预期效果    | 均匀浮点随机数生成受转换链路限制            | 浮点随机数生成吞吐提升，并增强上层分布采样底层供给能力          |

---

### 2.4 正态分布采样底层供给优化实现

该部分实现的功能，是为正态分布随机数生成提供更高效的底层随机源支撑，从而带动正态分布采样整体提速。

具体实现上，不主动替换原有正态分布算法，而是优先增强其底层均匀随机源供给，并对其中可安全批量化的局部转换步骤做局部优化。在调用链上保持分布采样接口与参数行为不变，将上层对随机 bit 或均匀浮点源的申请优先导向已优化的底层供给路径；当局部存在规则性强、不会破坏算法语义的转换步骤时，可附加批量化处理，否则保留 NumPy 原生流程。

该实现重点体现“底层加速、上层透明”的设计原则。若正态分布在实际验证中表现出收益不稳定或质量验证风险，则允许通过策略配置关闭该组合的优化路径并回退原生实现。

**落地示意：**

```text
优化前：每类分布直接依赖原生随机源

normal   ----\
weibull  -----+--> 原生随机源 --> 分布转换 --> 输出
binomial -----/
poisson  ----/


优化后：公共底层随机源先增强，再支撑上层分布

normal   ----\
weibull  -----+--> 增强后的底层随机源 --> 分布转换 --> 输出
binomial -----/            |
poisson  ----/             |
                           v
              可安全批量化的局部步骤同步增强
```

**原理解释：**

虽然示意图中展示了多个分布共享底层随机源的模式，但在本次 Benchmark 范围内，重点验证对象是 `time_normal_zig` 相关正态分布路径。正态分布生成通常基于底层均匀随机数，再通过特定算法转换为服从正态分布的样本。

该路径不能简单地把算法替换掉，因为正态分布的尾部概率、均值、方差等统计性质都依赖算法细节。如果算法改变，即使生成速度更快，也可能导致结果分布发生变化。因此，本 SR 的原则是：

```text
不改变分布算法
不改变 seed 与序列语义
不改变统计性质
只优化底层供给和安全局部步骤
```

这样既能降低风险，也能让底层随机源优化对正态分布路径产生收益。

**优化前后效果对比：**

| 对比项     | 优化前                       | 优化后                          |
| ------- | ------------------------- | ---------------------------- |
| 处理方式    | 正态分布采样直接依赖原生底层随机源         | 公共底层随机源优先复用增强后的批量供给路径        |
| 循环/分支开销 | 分布转换过程中持续拉取随机源，底层供给开销重复出现 | 底层随机源吞吐提升后，减少上层频繁拉取随机源的等待成本  |
| 访存/写回模式 | 随机源生成和分布转换之间可能存在更多中间处理    | 优化底层供给和安全局部步骤，减少不必要的中间开销     |
| 适用场景    | `normal` 相关批量采样路径         | 依赖大量均匀随机源或随机 bit 的批量正态分布采样场景 |
| 预期效果    | 正态分布采样受底层随机源效率限制          | 在不改变分布算法语义的前提下，正态分布采样获得链路收益  |

---

### 2.5 RNG 与统计链路联动验证实现

该部分实现的功能，是验证随机数优化路径不仅自身生成速度提升，而且进入统计链路后仍能保持正确、稳定和可解释。

随机数通常会被下游继续用于统计分析，例如：

```text
生成随机数组
   |
   +--> mean / std / var
   +--> percentile
   +--> histogram
   +--> max / min
```

因此，RNG 优化必须同时关注两类结果：

1. 随机数生成接口本身是否正确。
2. 随机数进入统计计算后，统计结果是否稳定、合理、与原生实现无显著差异。

实现上，联动验证使用固定 seed、多组 dtype、多组 size 和多种 BitGenerator 组合，生成随机数据后接入统计链路，验证均值、标准差、分位数、直方图分布等指标。该验证不替代专业随机性测试，但可作为工程验收中的回归保护手段。

**落地示意：**

```text
优化前：随机数性能与下游统计分开验证

随机数生成
   |
   v
只验证 RNG 性能


优化后：随机数生成与统计链路联合验证

随机数生成
   |
   v
+----------------------+
| mean / std / var     |
| percentile           |
| histogram            |
+----------------------+
   |
   v
验证统计特征是否稳定
```

**原理解释：**

只看随机数生成速度是不够的。随机数优化如果破坏了分布性质，可能在单个值上不容易看出来，但进入统计链路后会暴露问题。例如：

* 均值长期偏离理论值；
* 方差变小或变大；
* 直方图分布不均匀；
* 分位数位置异常；
* 不同 seed 下表现不稳定。

因此，联动验证的作用是把 RNG 优化从“单点性能优化”提升为“业务链路可用优化”。这对 SE 排查问题也很重要：如果 RNG 本身性能提升，但下游统计结果异常，就需要回到随机源、浮点转换或分布转换路径逐层定位。

**优化前后效果对比：**

| 对比项   | 优化前                | 优化后                                     |
| ----- | ------------------ | --------------------------------------- |
| 验证方式  | RNG 性能和统计链路通常分开看   | 将随机数生成结果接入统计链路做联动验证                     |
| 正确性保障 | 主要依赖单接口测试          | 增加均值、标准差、分位数、直方图等统计特征验证                 |
| 问题定位  | 随机数质量问题可能较晚暴露      | 可更早发现随机源、转换路径或分布路径异常                    |
| 适用场景  | 单独评估 RNG benchmark | 评估 `time_rng` 与 `time_statistics` 的组合链路 |
| 预期效果  | 只能证明单点接口更快         | 同时证明随机数优化后仍可支撑下游统计业务                    |

---

### 2.6 运行时选路、回退与验证实现

该部分实现的功能，是确保随机数优化路径只在适合的场景下启用，并在异常或不适配时能够平滑回退到原生实现，同时完成结果正确性和性能收益验证。

系统在运行时根据 CPU 能力、生成器类型、dtype、数据规模、内存布局以及策略开关选择实际执行路径。对于适配的热点路径优先尝试进入向量化优化实现；若数据量过小、当前组合未适配、环境变量显式关闭优化、执行过程中发生异常，或该组合已被标记为不稳定，则自动回退至 NumPy 原生实现。

验证方面，要求通过 NumPy 随机数相关测试，确保不同 seed、不同 dtype、不同 shape 下行为与原生实现一致；同时需对均值、方差、分位点、自相关、KS 检验、卡方检验等指标进行验证，确保优化后随机数在统计性质上不存在显著劣化；并结合主性能集和联动验证集评估整体链路表现。

**落地示意：**

```text
随机数请求
   |
   v
+--------------------------+
| 读取请求特征             |
| generator / dtype / size |
| distribution / shape     |
+--------------------------+
   |
   v
+--------------------------+
| 判断是否适合增强路径     |
| 平台能力 / 数据规模      |
| dtype / 分布类型 / 开关  |
+--------------------------+
   |
   +---- 不适合 / 关闭 / 风险组合 ----> NumPy 原生路径
   |
  适合
   |
   v
+--------------------------+
| 进入向量化优化路径       |
+--------------------------+
   |
   +---- 异常 / 校验失败 ----> NumPy 原生路径
   |
  成功
   |
   v
+--------------------------+
| 输出结果并记录路径信息   |
+--------------------------+
```

**原理解释：**

优化路径不是越多越好。对于小数组，进入向量化路径本身可能需要额外判断、准备和收尾，反而比原生路径更慢。对于复杂 dtype、复杂 shape 或尚未验证充分的组合，如果强行启用优化路径，可能带来正确性风险。

因此，本 SR 采用“主路径增强 + 风险场景回退”的策略：

```text
大数组、连续内存、常见 dtype、已验证组合
        -> 启用优化路径

小数组、复杂组合、风险场景、异常场景
        -> 回退 NumPy 原生路径
```

这样可以确保性能收益集中在高价值场景，同时保持整体行为稳定。路径信息记录也很重要，它可以帮助 SE 判断某个 benchmark 是否命中了优化路径，或者为什么回退到了原生路径。

**优化前后效果对比：**

| 对比项     | 优化前                         | 优化后                                |
| ------- | --------------------------- | ---------------------------------- |
| 处理方式    | 主要依赖原生实现，缺少平台感知选路           | 根据平台能力、dtype、规模、分布类型等动态选择增强路径或原生路径 |
| 循环/分支开销 | 不区分小规模和大规模请求，难以避免不必要的增强路径开销 | 通过阈值和策略控制，仅在收益明确的场景启用优化            |
| 访存/写回模式 | 不针对不同内存布局做差异化处理             | 根据连续性、数据规模和输出形态选择更合适的执行路径          |
| 适用场景    | 所有场景统一走原生稳定路径               | 主路径增强，复杂场景、小数据量和异常场景回退             |
| 预期效果    | 性能对比和问题定位依赖人工分析             | 形成“可启用、可回退、可验证、可比较”的完整闭环           |

---

# 二、Statistics 性能优化 SR — 功能实现设计

## 0 SR 描述、输入与输出

**SR 描述：**

本 SR 的功能是优化 NumPy 在鲲鹏 920B / Arm 平台上的统计计算执行路径。它不改变 `sum`、`mean`、`var`、`std`、`max`、`argmax`、`percentile`、`histogram` 等接口的 Python API，不改变统计定义、dtype 提升规则、NaN 规则、边界行为和数值可靠性；它要做的是在适合的连续内存、常见 dtype、大数组或高频调用场景下，对底层归约、比较、分桶、过滤和函数入口分发进行平台化增强。

可以把这个 SR 理解为：

```text
用户仍然调用 NumPy 原有统计接口
        |
        v
NumPy 保持原有统计公式、边界规则和返回格式
        |
        v
在适合的 Arm 场景下，扫描、归约、比较、分桶过程执行得更快
```

**输入：**

Statistics SR 的输入是一次“统计计算请求”。这个请求通常由输入数组、统计函数参数和运行时平台条件共同决定。

| 输入类型 | 具体内容 | 说明 |
| --- | --- | --- |
| 输入数据 | `ndarray` 或 array-like 转换后的数组 | 包含 shape、dtype、内存布局、stride、是否连续、是否含 NaN/Inf |
| 统计接口 | `sum`、`mean`、`var`、`std`、`max`、`argmax`、`percentile`、`histogram` 等 | 决定是归约、比较、选择、插值还是分桶路径 |
| 轴与形状参数 | `axis`、`keepdims`、输入维度、归约维度长度 | 决定是否能走连续内存和规则 axis 快路径 |
| dtype 相关参数 | 输入 dtype、`dtype=`、输出 dtype、类型提升规则 | 决定累计类型、比较类型和返回类型 |
| 输出参数 | `out`、输出数组布局 | 决定是否能连续写回，以及是否需要额外拷贝 |
| 统计语义参数 | `ddof` / `correction`、`where`、`method`、`q`、`bins`、`range`、`weights`、`density` | 决定 `var/std`、`percentile`、`histogram` 等复杂路径的语义 |
| 平台能力 | Arm NEON/SVE 支持、CPU dispatch 结果、运行时开关 | 决定是否启用鲲鹏平台增强路径 |

对应到 benchmark，可以理解为：`mean/sum/max/std/var` 主要输入是数组 dtype、shape、axis 和连续性；`percentile` 还需要关注 `q`、method、NaN 和临时数组；`histogram` 还需要关注 `bins`、`range`、`weights` 和桶分布。

**输出：**

Statistics SR 的功能输出仍然是 NumPy 原统计接口返回的结果，不新增对用户可见的新返回值。

| 输出类型 | 具体内容 | 约束 |
| --- | --- | --- |
| 统计结果 | 标量或 `ndarray`，例如 `sum/mean/var/std/max` 的结果 | shape、dtype、精度和 NumPy 语义必须保持一致 |
| 位置结果 | `argmax/argmin/nanargmax/nanargmin` 返回的索引 | 相等值时返回首次出现位置，NaN 和全 NaN 行为必须保持一致 |
| 分位数结果 | `percentile/quantile/nanpercentile` 返回的标量或数组 | selection、interpolation、method、NaN 处理语义必须保持一致 |
| 直方图结果 | `histogram` / `histogramdd` 返回的计数和 bin edges | bin 边界、最后一个 bin 右闭、weights、density 规则必须保持一致 |
| 内部路径结果 | 是否命中 Arm 优化路径、是否回退原生路径 | 这是开发和验证信息，不改变 Python API |
| 性能结果 | benchmark 中体现为执行时间下降，perf 中体现为热点变化 | 性能收益必须由 benchmark 和 perf 证据支撑 |

因此，本 SR 的验收重点不是“输出新的统计结果”，而是：**同样的输入数组和参数，统计结果完全兼容 NumPy 原语义，但在鲲鹏 920B 上热点路径更快**。

## 1 实现思路

本 SR 面向鲲鹏 920B 平台，对 NumPy 常用统计运算路径进行性能优化。在**不改变上层 Python API 语义、不改变统计计算结果定义、不破坏数值可靠性和边界行为**的前提下，利用 Arm 平台可用的 NEON/SVE 向量能力、连续访存能力和平台感知分发能力，对统计运算中的热点基础路径进行增强，从而提升大数组统计分析场景的执行效率。

本 SR 不重写 NumPy 全部 statistics 逻辑，而是围绕底层最耗时、最规则、最适合向量化的执行部分进行增强。整体优化重点包括：

* 基础归约统计：`sum`、`mean`。
* 离散程度统计：`var`、`std`。
* 最值与位置统计：`max`、`argmax`、`nanargmax`、`nanargmin`。
* 分位数相关统计：`percentile`、`nanpercentile`。
* 直方图统计：`histogram1d`、`histogram2d`。
* 综合统计链路：`bench_xiecheng.RandomAndStats.time_statistics(4000)`。

统计运算的核心特点是：大量数据被反复扫描、比较、累加或分桶。因此，在大数组、连续内存、规则 axis 的场景下，批量加载、批量计算、局部累计和块间合并通常能够带来较明确收益。

为了方便开发判断优化层级，可以先把这些用例按“算子形态”理解：

```text
sum / mean / max:
    主要是扫描数组，把很多元素合并成少量结果

std / var:
    先算 mean，再算每个元素离 mean 多远，最后再归约

argmax / nanargmax:
    一边比较值，一边记住位置，还要处理相等值和 NaN 规则

percentile / nanpercentile:
    不是简单扫描，需要找到排序后某个位置的元素，可能还要插值

histogram / histogramdd:
    每个输入值先映射到桶编号，再对桶计数
```

因此，不同用例的优化方式不同。`sum`、`mean`、`max` 更接近底层扫描内核优化；`percentile`、`histogram` 则有较多函数层逻辑、临时数组和数据搬运，不能只用“向量化归约”一类方案概括。

面向“Arm 追平 Zen4”，Statistics 的设计也不应该写成“所有统计函数都 SIMD 化”。更准确的分类是：

```text
算子层优化：
    对 sum / mean / max / var / std / argmax 这类扫描型算子补 Arm 专用内核、
    多累加器、连续访存和 dtype 专用路径。

函数层优化：
    对 percentile / histogram / nan* 这类 Python 层逻辑较重的函数，
    优先减少临时数组、分发、过滤、桶索引和数据搬运成本。
```

这两类都可能帮助 Arm 追 Zen4，但抓手不同。算子层主要看底层循环、SIMD、访存和依赖链；函数层主要看上层调用链、临时数组、`partition` / `bincount` / `searchsorted` 周边成本。如果 perf 热点落在 Python 包装、临时数组构造或外部库里，就不能把根因简单归到 NumPy 的 SIMD 内核。

---

### 1.1 归约类统计路径优化

这一类功能主要负责完成数组元素的累加和整体聚合，是 `sum`、`mean` 等统计接口的基础执行路径。本质上，它解决的是“对大批量数据进行重复归约运算”的问题。

归约可以理解为把一批数据合并成一个或一组结果，例如：

```text
sum:   x0 + x1 + x2 + ... + xn
mean: (x0 + x1 + x2 + ... + xn) / n
```

其核心思路是把逐元素累加改为批量加载、批量累积和分块合并，从而减少循环控制开销、缩短依赖链，并提升连续访问场景下的 cache 利用率。

这类路径规则性强、覆盖面广，也是统计运算中最容易通过向量化获得收益的部分。对小白来说，可以把它理解成“以前一次只拿一个数相加，现在一次拿一组数分别累加，最后再把几组小结果合起来”。

这类用例同时包含两种优化空间：

```text
函数层优化：
    对小数组更明显，例如减少 np.mean / arr.mean 的参数处理和分发开销

底层内核优化：
    对大数组更明显，例如连续加载、SIMD 累加、多累加器减少依赖链
```

例如 `bench_core.StatsMethods.time_mean(..., 100)` 更容易看到函数入口和小数组开销，`time_mean(..., 10000)`、`bench_function_base.Mean.time_mean(100000)` 更容易看到底层扫描内核的吞吐能力。

**相关用例：**

* `bench_core.StatsMethods.time_sum('float64', 100)`
* `bench_core.StatsMethods.time_sum('float64', 10000)`
* `bench_core.StatsMethods.time_sum('int64', 100)`
* `bench_core.StatsMethods.time_sum('int64', 10000)`
* `bench_core.StatsMethods.time_mean('float32', 100)`
* `bench_core.StatsMethods.time_mean('float32', 10000)`
* `bench_core.StatsMethods.time_mean('float64', 100)`
* `bench_core.StatsMethods.time_mean('float64', 10000)`
* `bench_core.StatsMethods.time_mean('int64', 100)`
* `bench_core.StatsMethods.time_mean('int64', 10000)`
* `bench_core.StatsMethods.time_mean('uint64', 100)`
* `bench_core.StatsMethods.time_mean('uint64', 10000)`
* `bench_reduce.StatsReductions.time_mean('bool_')`
* `bench_reduce.StatsReductions.time_mean('complex64')`
* `bench_reduce.StatsReductions.time_mean('float32')`
* `bench_reduce.StatsReductions.time_mean('float64')`
* `bench_reduce.StatsReductions.time_mean('int64')`
* `bench_reduce.StatsReductions.time_mean('uint64')`
* `bench_function_base.Mean.time_mean(1)`
* `bench_function_base.Mean.time_mean(10)`
* `bench_function_base.Mean.time_mean(100000)`
* `bench_function_base.Mean.time_mean_axis(1)`
* `bench_function_base.Mean.time_mean_axis(10)`
* `bench_function_base.Mean.time_mean_axis(100000)`

---

### 1.2 最值与位置统计路径优化

这一类功能主要负责从数组中找出最大值以及对应的位置，是 `max`、`argmax`、`nanargmax`、`nanargmin` 等接口的基础能力。本质上，它解决的是“大规模扫描 + 比较 + 位置跟踪”的问题。

其思路是对值比较采用批量向量比较，对局部最值和局部位置同时维护，再在块间进行归并。该类路径的主要收益来自减少逐元素比较和分支判断次数。由于最值统计本质上是规则扫描型场景，比较适合做向量化处理，因此在大数组上通常能得到比较稳定的收益。

对于 `argmax` 这类位置统计，优化不能只关注最大值本身，还必须保持索引语义一致。例如当多个元素具有相同最大值时，NumPy 对返回哪个位置有明确行为，优化路径必须保持一致。

可以把 `max` 和 `argmax` 的区别理解成：

```text
max:
    只需要知道最大值是多少

argmax:
    既要知道最大值是多少，还要知道它第一次出现在哪里
```

`nanargmax`、`nanargmin` 还要额外处理 NaN。NaN 的语义不是简单跳过所有异常值这么粗糙：全 NaN、部分 NaN、不同 dtype 和边界值都要保持 NumPy 原有行为。因此这类优化不能只看比较指令是否更快，还要看分支布局、NaN 检测和错误路径是否仍然正确。

**相关用例：**

* `bench_core.StatsMethods.time_max('float64', 100)`
* `bench_core.StatsMethods.time_max('float64', 10000)`
* `bench_core.StatsMethods.time_max('int64', 100)`
* `bench_core.StatsMethods.time_max('int64', 10000)`
* `bench_reduce.FMinMax.time_max(<class 'numpy.float64'>)`
* `bench_reduce.StatsReductions.time_max('float64')`
* `bench_reduce.StatsReductions.time_max('int64')`
* `bench_reduce.ArgMax.time_argmax(<class 'bool'>)`
* `bench_reduce.ArgMax.time_argmax(<class 'numpy.float32'>)`
* `bench_reduce.ArgMax.time_argmax(<class 'numpy.float64'>)`
* `bench_reduce.ArgMax.time_argmax(<class 'numpy.int16'>)`
* `bench_reduce.ArgMax.time_argmax(<class 'numpy.int32'>)`
* `bench_reduce.ArgMax.time_argmax(<class 'numpy.int64'>)`
* `bench_reduce.ArgMax.time_argmax(<class 'numpy.int8'>)`
* `bench_reduce.ArgMax.time_argmax(<class 'numpy.uint16'>)`
* `bench_reduce.ArgMax.time_argmax(<class 'numpy.uint32'>)`
* `bench_reduce.ArgMax.time_argmax(<class 'numpy.uint64'>)`
* `bench_reduce.ArgMax.time_argmax(<class 'numpy.uint8'>)`
* `bench_lib.Nan.time_nanargmax(200, 0)`
* `bench_lib.Nan.time_nanargmax(200000, 0)`
* `bench_lib.Nan.time_nanargmin(200, 0)`
* `bench_lib.Nan.time_nanargmin(200000, 0)`

---

### 1.3 方差和标准差路径优化

这一类功能主要负责衡量数据离散程度，是 `var`、`std` 等接口的核心执行路径。本质上，它既包含均值相关计算，也包含偏差平方和累计，因此比普通归约更复杂、对数值稳定性要求更高。

方差和标准差可以简化理解为：

```text
mean = 数据平均值
var  = 每个元素与 mean 的差的平方，再求平均
std  = var 的平方根
```

它们通常需要多阶段处理：

```text
第一阶段：计算 mean
第二阶段：计算 (x - mean)^2 的累积和
第三阶段：计算 var / std
```

本 SR 的思路是在保证数值可靠性的前提下，优先增强均值计算阶段和偏差平方和累计阶段的批量处理能力。这类路径的收益主要来自减少多轮遍历中的标量计算和重复访存，但由于其对浮点误差更敏感，因此实现策略要比 `mean` 更保守，必须在吞吐提升和数值稳定之间取得平衡。

这里有一个容易忽略的点：`var` / `std` 不是“把公式写成一行就可以”。如果直接用 `mean(x*x) - mean(x)^2`，在一些数据范围下可能产生更大的浮点误差。NumPy 当前实现更接近“两遍计算”：先求均值，再求偏差平方和。这样多扫一次数组，但数值行为更稳。

从优化层级看：

```text
函数层：
    dtype、axis、where、ddof、mean 参数会影响路径选择，小数组会受这些开销影响

底层层面：
    mean 阶段、偏差平方阶段、sum 阶段都可能成为热点
```

因此该类路径可以优化，但不能为了少扫一遍数组随意换公式。尤其是 `int`、`uint`、`bool` 默认可能转成 `float64` 计算，`complex` 还有实部/虚部平方和语义，这些都属于必须保留的源码行为。

**相关用例：**

* `bench_core.StatsMethods.time_var('float64', 100)`
* `bench_core.StatsMethods.time_var('float64', 10000)`
* `bench_core.StatsMethods.time_var('int64', 100)`
* `bench_core.StatsMethods.time_var('int64', 10000)`
* `bench_core.StatsMethods.time_std('bool_', 100)`
* `bench_core.StatsMethods.time_std('bool_', 10000)`
* `bench_core.StatsMethods.time_std('complex64', 100)`
* `bench_core.StatsMethods.time_std('complex64', 10000)`
* `bench_core.StatsMethods.time_std('float32', 100)`
* `bench_core.StatsMethods.time_std('float32', 10000)`
* `bench_core.StatsMethods.time_std('float64', 100)`
* `bench_core.StatsMethods.time_std('float64', 10000)`
* `bench_core.StatsMethods.time_std('int64', 100)`
* `bench_core.StatsMethods.time_std('int64', 10000)`
* `bench_core.StatsMethods.time_std('uint64', 100)`
* `bench_core.StatsMethods.time_std('uint64', 10000)`
* `bench_reduce.StatsReductions.time_std('bool_')`
* `bench_reduce.StatsReductions.time_std('complex64')`
* `bench_reduce.StatsReductions.time_std('float32')`
* `bench_reduce.StatsReductions.time_std('float64')`
* `bench_reduce.StatsReductions.time_std('int64')`
* `bench_reduce.StatsReductions.time_std('uint64')`

---

### 1.4 分位数与 NaN 分位数路径优化

这一类功能主要负责描述数据分布中的位置统计，是 `percentile`、`nanpercentile` 等接口的核心基础能力。本质上，它涉及局部排序、选择、分区和数据搬运等过程。

分位数不是简单扫描就能完成的统计。它通常需要找到排序后某个位置的元素，或者在两个位置之间做插值。因此，它比 `mean`、`sum` 更复杂，也更不适合整体重写。

本 SR 的设计重点是优化规则性强、收益明确的基础步骤，例如连续数据块预处理、局部比较、数据搬运、中间缓冲处理以及 NaN 数据过滤相关流程。复杂的排序、选择和插值语义保持原生实现，避免引入结果偏差。

对小白来说，分位数可以这样理解：`mean` 关心“所有数加起来平均是多少”，而 `percentile` 关心“把所有数从小到大排好后，某个位置上的数是多少”。例如 50 分位数就是中位数附近的位置，25/75 分位数就是四分位数。

因此它的热点和 `sum` 完全不同：

```text
sum / mean:
    扫描一遍，边扫边累加

percentile:
    找排序位置，可能 partition，可能取相邻两个值做插值

nanpercentile:
    还要先处理 NaN，再做分位数逻辑
```

这类用例有函数层优化价值，例如减少临时数组、减少重复转换、优化小数组路径；但真正的大数组耗时通常会落在 partition、排序相关选择、NaN 过滤和数据搬运上。因此文档里不应把它写成普通向量归约。

**相关用例：**

* `bench_function_base.Percentile.time_percentile`
* `bench_function_base.Percentile.time_percentile_small`
* `bench_function_base.Percentile.time_quartile`
* `bench_lib.Nan.time_nanpercentile(200, 0)`
* `bench_lib.Nan.time_nanpercentile(200, 0.1)`
* `bench_lib.Nan.time_nanpercentile(200, 2.0)`
* `bench_lib.Nan.time_nanpercentile(200, 50.0)`
* `bench_lib.Nan.time_nanpercentile(200, 90.0)`
* `bench_lib.Nan.time_nanpercentile(200000, 0)`
* `bench_lib.Nan.time_nanpercentile(200000, 0.1)`
* `bench_lib.Nan.time_nanpercentile(200000, 2.0)`
* `bench_lib.Nan.time_nanpercentile(200000, 50.0)`
* `bench_lib.Nan.time_nanpercentile(200000, 90.0)`

---

### 1.5 Histogram 统计分桶路径优化

这一类功能主要负责根据数值范围对数据进行分桶统计，是 `histogram`、`histogram2d` 等接口的基础能力。本质上，它解决的是“桶索引计算 + 分桶累计 + 结果合并”的问题。

直方图计算可以理解为：

```text
输入数据 x
   |
   v
判断 x 落在哪个区间
   |
   v
对应 bucket 计数 +1
```

在大数组场景下，瓶颈往往不只是算术本身，还包括桶索引计算和桶计数写回。尤其当很多数据落到相同 bucket 时，会出现热点写入。通过局部统计后再合并写回，可以提升局部性并降低热点更新带来的性能损耗。

直方图和 `sum` / `mean` 的区别在于，它不是把所有数合成一个结果，而是把每个输入分配到不同桶里。每个元素大概会经历三步：

```text
1. 判断是否在统计范围内
2. 根据数值计算桶编号
3. 给对应桶的计数加一
```

`histogram` 的 1D 等宽分桶通常更容易优化，因为桶编号可以通过线性公式算出来；`histogramdd` 需要多个维度分别找桶，再把多维桶编号映射成一维索引，逻辑更复杂。函数层可以减少边界检查、临时数组和分发开销，但大数组的核心热点通常在桶索引计算、过滤、`bincount` 和写回冲突上。

**相关用例：**

* `bench_function_base.Histogram1D.time_fine_binning`
* `bench_function_base.Histogram1D.time_full_coverage`
* `bench_function_base.Histogram1D.time_small_coverage`
* `bench_function_base.Histogram2D.time_fine_binning`
* `bench_function_base.Histogram2D.time_full_coverage`
* `bench_function_base.Histogram2D.time_small_coverage`

---

### 1.6 综合统计链路优化与保守回退

这一类功能主要负责在复杂 axis、复杂 stride、非连续内存、小数据量、NaN/Inf 边界值和复杂 dtype 场景下保障功能正确性和执行稳定性。它本质上解决的是“哪些场景适合优化，哪些场景应保持原生实现”的问题。

本 SR 不追求所有统计场景强行走优化路径，而是将收益最明确的规则场景作为主优化范围，对复杂场景保持原生实现或动态回退。这样做的原因是统计运算边界情况很多，如果盲目扩展优化范围，容易引入精度问题、边界错误或性能倒挂。

这一类设计的重点不在于新增统计能力，而在于保障增强路径的启用边界可控、异常可回退、结果可验证。

可以把回退策略理解成“只在确定能跑得更快且结果一致的场景启用优化路径”。例如连续内存、常见 dtype、简单 axis、大数组通常适合增强；非连续 stride、复杂 axis、object dtype、小数组、NaN/Inf 密集场景可能让优化路径收益变小甚至变慢。对这些情况保留原生实现，不是性能目标失败，而是为了控制风险。

**相关用例：**

* `bench_xiecheng.RandomAndStats.time_statistics(4000)`
* `bench_core.StatsMethods.time_mean('float64', 10000)`
* `bench_core.StatsMethods.time_std('float64', 10000)`
* `bench_core.StatsMethods.time_var('float64', 10000)`
* `bench_core.StatsMethods.time_sum('float64', 10000)`
* `bench_core.StatsMethods.time_max('float64', 10000)`
* `bench_reduce.StatsReductions.time_mean('float64')`
* `bench_reduce.StatsReductions.time_std('float64')`
* `bench_reduce.StatsReductions.time_max('float64')`
* `bench_function_base.Mean.time_mean(100000)`
* `bench_function_base.Mean.time_mean_axis(100000)`
* `bench_function_base.Percentile.time_percentile`
* `bench_function_base.Histogram1D.time_full_coverage`
* `bench_function_base.Histogram2D.time_full_coverage`

---

### 1.7 Statistics 主要优化方法与原理

Statistics SR 的核心方法不是单一的“向量化”，而是按算子特点选择不同的实现增强方式。`sum`、`mean`、`max` 是扫描归约；`std`、`var` 是多阶段归约；`argmax` 要同时维护值和位置；`percentile` 依赖选择和分区；`histogram` 依赖桶索引和计数写回。因此各类用例需要不同方法。

这里同样需要校正分类边界：这些方法不是全部对应独立源码文件。例如 `mean` 会调用求和归约后再除以元素个数；`std` 会复用 `var`；`histogram` 内部会走过滤、桶编号和 `bincount`。所以功能设计中应按“主要算法方法”说明，而不是把每个 benchmark 名字当成单独模块。

源码依据如下：

| 源码入口 | 对应功能 | 支撑的优化方法 |
| --- | --- | --- |
| `numpy/_core/_methods.py::_mean` | `mean` 入口，内部依赖 `umr_sum` 后做除法 | 函数入口 fast path、连续内存归约、dtype 专用路径 |
| `numpy/_core/_methods.py::_var` | `var` 入口，先算 mean，再算偏差平方和 | `var/std` 两阶段归约增强 |
| `numpy/_core/_methods.py::_std` | `std` 入口，调用 `_var` 后开平方 | `var/std` 两阶段归约增强 |
| `numpy/_core/_methods.py::_amax/_amin` | `max/min` 归约入口 | 连续内存归约向量化、dtype 专用路径 |
| `numpy/_core/src/umath/loops_utils.h.src::*_pairwise_sum` | 浮点求和使用 pairwise sum 和多累加器块内求和 | `sum/mean/var/std` 连续归约增强，保持数值语义 |
| `numpy/_core/src/umath/loops_minmax.dispatch.c.src::simd_reduce_c_*` | `min/max` 已有基于 `NPY_SIMD` 的连续 reduce 模板 | Arm NEON/SVE min/max 路径补强 |
| `numpy/_core/src/multiarray/argfunc.dispatch.c.src` | `argmax/argmin` 的 dtype/CPU dispatch 路径 | 最值与位置联合维护、first occurrence 语义 |
| `numpy/lib/_function_base_impl.py::_quantile` | `percentile/quantile` 核心，包含 partition、take、interpolation | 分位数局部选择优化 |
| `numpy/lib/_nanfunctions_impl.py` 中 `nan*` 系列 | NaN 统计接口 | NaN 检测、过滤、错误路径保持 |
| `numpy/lib/_histograms_impl.py::histogram` | 1D histogram，包含 range 过滤、bin index、bincount | Histogram 分桶索引与局部累计 |
| `numpy/lib/_histograms_impl.py::histogramdd` | 多维 histogram，包含 searchsorted、ravel_multi_index、bincount | 多维分桶索引优化 |
| `numpy/_core/src/multiarray/compiled_base.c::arr_bincount` | `bincount` 底层计数实现，被 histogram 快路径调用 | 桶计数局部累计和热点写回优化 |

#### 面向 Arm 追平 Zen4 的 Statistics 优化重点

现有七类方法总体方向正确，但需要把“哪些是算子层、哪些是函数层”写清楚，否则开发容易把所有用例都套成一种 SIMD reduce。建议按下表作为真正的实现设计依据。

| 优化方法 | Arm 追平 Zen4 的主要抓手 | 源码依据与边界 |
| --- | --- | --- |
| 函数入口 fast path | 小数组或高频调用时，减少 `_mean/_var/_std` 的 axis、where、dtype、out 分发成本；避免还没进入 C 内核就输给 Zen4 | 只适合默认参数、连续数组、常见 dtype；复杂 axis、where、subclass、object dtype 必须回退 |
| 连续内存归约向量化 | 对 `sum/mean/max` 建 Arm NEON/SVE 友好的多累加器循环，降低单 accumulator 依赖链；连续 load + 块内 reduce + 尾部标量 | NumPy 已有 `*_pairwise_sum` 保精度，Arm 优化不能把 pairwise 语义改成随意重排的快速求和 |
| dtype 专用归约路径 | 为 `float32/float64/int64/uint64/bool_` 固定 dtype 减少运行时分支，让编译器和 `NPY_SIMD` 选到更稳定的 Arm 指令 | int/bool/complex 的类型提升和溢出规则不同，不能为了统一 SIMD 内核混成一种累计类型 |
| `var/std` 两阶段增强 | 分别优化 mean 阶段和偏差平方和阶段；第二阶段对 `(x - mean)`、平方、累计做块内多累加器 | 不能改成 `mean(x*x) - mean(x)^2`，否则数值误差和 NumPy 当前语义可能变化 |
| 最值与位置联合维护 | 对 `argmax/nanargmax` 做向量比较时同步维护 lane index；块间合并保持“相等时返回最早位置” | 只做值比较会破坏 argmax 语义；NaN、全 NaN 和 first occurrence 是回归风险点 |
| 分位数局部选择优化 | 重点不在 SIMD reduce，而在减少复制、NaN 过滤、partition 周边取数和插值的数据搬运 | `_quantile` 依赖 partition/take/interpolation；不能把 percentile 写成排序或普通归约 |
| Histogram 分桶优化 | 1D 等宽 bin 优先优化 range 过滤、bin index 计算和 `bincount` 前临时索引；热点桶多时用局部 histogram 缓冲减少全局写冲突 | `histogram` 快路径调用 `np.bincount`；最后一个 bin 右边界包含规则、weights、density 语义必须保持 |

真正针对 Arm 的优先级建议是：先做连续内存、常见 dtype、大数组路径，因为这类最容易把 Kunpeng 的访存和 NEON/SVE 能力用起来；再做 histogram 的等宽 1D 快路径和 `bincount` 周边，因为它常被桶写回和临时数组拖慢；最后再考虑 percentile，因为它的核心是选择和插值，不是规则归约，盲目 SIMD 化收益和风险都不稳定。

#### 方法一：函数入口 fast path

该方法面向小数组和默认参数场景，例如 `bench_reduce.StatsReductions.time_mean`、`bench_function_base.Mean.time_mean(1)`、`time_mean(10)`。这类用例数据量很小，真正计算只需要很少指令，函数入口的参数解析、dtype 判断、axis/where 处理和 Python 到 C 的分发成本会占较大比例。

优化原理是为最常见组合建立短路径：

```text
默认 axis / 默认 where / 常见 dtype / 连续数组
        |
        v
跳过复杂通用分发
        |
        v
直接进入对应底层实现
```

该方法不会改变统计公式，只是减少进入底层计算前的通用检查和分发成本。它适合小数组和高频调用；对 10 万级大数组，收益通常会被底层扫描成本淹没。

#### 方法二：连续内存归约向量化

该方法面向 `sum`、`mean`、`max` 等基础归约，覆盖 `bench_core.StatsMethods.time_sum`、`time_mean`、`time_max`，以及 `bench_function_base.Mean.time_mean(100000)` 等大数组用例。

归约的本质是把很多输入合成少量输出。普通标量循环容易形成长依赖链：

```text
acc = acc + x0
acc = acc + x1
acc = acc + x2
...
```

每一步都依赖上一步的 `acc`。优化方法是分块加载，并使用多个局部累加器或向量寄存器并行累计：

```text
acc0 累加 x0, x4, x8 ...
acc1 累加 x1, x5, x9 ...
acc2 累加 x2, x6, x10 ...
acc3 累加 x3, x7, x11 ...
最后合并 acc0..acc3
```

这样可以缩短单条依赖链，提高流水线和 SIMD 利用率。对连续内存、常见 dtype、大数组最有效；对非连续 stride、复杂 axis 或 object dtype 应保持原生路径。

#### 方法三：dtype 专用归约路径

该方法面向 `float32`、`float64`、`int64`、`uint64`、`bool_` 等常见 dtype。不同 dtype 的统计语义不同，不能完全共用一套实现。例如 `mean(int64)` 默认结果需要按 NumPy 规则提升到 `float64`；`bool_` 的求和和均值也有自己的转换行为；`complex` 的 `std` / `var` 还涉及实部和虚部的平方和。

优化原理是对常见 dtype 建立专用内核：

```text
float32:
    使用适合 float32 的加载、累加和转换规则

float64:
    使用 float64 累加和向量化路径

int / uint / bool:
    保留 NumPy dtype 提升规则，再进入对应累计路径

complex:
    保留复数统计语义，不能当成普通浮点数组直接处理
```

该方法的收益来自减少运行时 dtype 分支，并让编译器或手写内核针对固定 dtype 做更好的指令选择。

#### 方法四：`var` / `std` 两阶段归约增强

该方法面向 `bench_core.StatsMethods.time_var`、`time_std` 和 `bench_reduce.StatsReductions.time_std`。`var` / `std` 不是一次简单扫描，通常包含：

```text
第一阶段：求 mean
第二阶段：求 (x - mean)^2 的和
第三阶段：除以计数，std 再开平方
```

优化原理是分别增强两个扫描阶段，而不是替换统计公式。第一阶段复用 `sum/mean` 的批量归约能力；第二阶段对 `(x - mean)`、平方和累计做批量处理，减少重复标量计算和访存开销。

这里不能为了少扫一遍数组随意改成：

```text
var = mean(x*x) - mean(x)^2
```

这个公式虽然看起来少了一些步骤，但在数值接近、数据范围较大时容易产生更明显的浮点误差。功能实现设计应明确：优化执行方式，不改变 NumPy 当前数值语义。

#### 方法五：最值与位置联合维护

该方法面向 `max`、`argmax`、`nanargmax`、`nanargmin`。`max` 只要值，`argmax` 同时要值和位置。优化 `argmax` 时不能只找最大值，还要保证返回位置符合 NumPy 语义。

原理如下：

```text
每个块内：
    找局部最大值
    同时记录局部最大值第一次出现的位置

块间合并：
    比较局部最大值
    如果值相等，保留更靠前的位置
```

`nanargmax` / `nanargmin` 还需要处理 NaN。NaN 判断可以通过批量 mask 降低分支成本，但全 NaN、部分 NaN、错误路径必须保持原行为。

#### 方法六：分位数局部选择优化

该方法面向 `bench_function_base.Percentile.time_percentile`、`time_quartile`、`time_percentile_small` 和 `bench_lib.Nan.time_nanpercentile`。

分位数不是扫描归约。它需要找到排序后某个位置的值，或者找到相邻两个位置后做插值。NumPy 通常不需要完整排序所有数据，而是通过 partition 找到目标位置附近的元素。

优化原理是减少分位数链路中的局部成本：

```text
1. 减少输入复制和临时数组
2. 对小数组减少函数层开销
3. 对 NaN 数据先做高效过滤或检测
4. 保持 partition、take、interpolation 的原有语义
```

该方法不适合写成普通 SIMD reduce。它的核心不是“所有元素加起来”，而是“找到排序位置并按方法插值”。

#### 方法七：Histogram 分桶索引与局部累计

该方法面向 `Histogram1D.time_full_coverage`、`time_small_coverage`、`time_fine_binning` 和 `Histogram2D.*`。

直方图的核心不是累加所有值，而是把每个输入映射到桶：

```text
输入值 x
    -> 判断是否在 range 内
    -> 计算 bin index
    -> hist[bin index] += 1
```

1D 等宽分桶可以用线性公式快速算桶编号，适合优化范围过滤、bin index 计算和 `bincount` 前的临时数组处理。2D / ND histogram 需要多个维度分别找桶，再映射成扁平索引，复杂度更高。

优化方法包括：

```text
1. 对 1D 等宽 bin 建立快速桶编号路径
2. 分块处理输入，降低临时数组峰值
3. 使用局部 histogram 缓冲，再合并到全局结果
4. 对小 coverage 场景尽早过滤范围外数据
```

局部 histogram 的原理是减少多个元素同时写同一个全局桶造成的热点写回，提高缓存局部性。该方法必须保持边界语义，例如最后一个 bin 是否包含右边界。

#### Statistics 方法与用例对应关系

| 主要方法 | 覆盖用例 | 主要收益来源 |
| --- | --- | --- |
| 函数入口 fast path | 小数组 `StatsReductions`、`Mean.time_mean(1/10)` | 减少参数解析、dtype/axis/where 分发 |
| 连续内存归约向量化 | `sum`、`mean`、`max` 大数组 | 批量加载、多累加器、SIMD、连续访存 |
| dtype 专用归约路径 | `float32`、`float64`、`int64`、`uint64`、`bool_` | 减少 dtype 分支，保留类型提升规则 |
| `var` / `std` 两阶段增强 | `StatsMethods.time_var/std`、`StatsReductions.time_std` | 分别优化 mean 和偏差平方和阶段 |
| 最值与位置联合维护 | `argmax`、`nanargmax`、`nanargmin` | 批量比较，同时维护索引和 NaN 语义 |
| 分位数局部选择优化 | `Percentile.*`、`Nan.time_nanpercentile` | 减少临时数组、NaN 处理、partition 周边开销 |
| Histogram 分桶优化 | `Histogram1D.*`、`Histogram2D.*` | 优化范围过滤、桶索引、局部累计和合并 |

---

## 2 实现设计

基于上述思路，Statistics SR 的实现按各类统计优化路径分别说明具体做法。实现设计中重点说明每类优化前后的执行差异、收益来源、风险边界和回退策略。

---

### 2.1 归约类统计优化实现

该部分实现的功能，是提升 `sum`、`mean` 等基础归约路径的执行效率，为大数组聚合计算提供更高效的底层执行能力。

对于连续数组和规则 axis 场景，采用分块遍历、批量加载和局部累积的方式处理输入数据，在块内先完成批量级归约，再在块间进行合并；当数据规模不足阈值或输入布局不适合向量化时，直接走原生路径。

对于 `mean`，在完成批量求和后按 NumPy 原有规则完成元素个数归一化；对于 axis 归约场景，采用“块内归约 + 块间合并”的方式控制中间结果规模。该实现主要提升大批量归约类统计路径的吞吐。

**落地示意：**

```text
优化前：逐元素累计

输入数组 A[0..N)
        |
        v
+----------------------+
| acc = acc + A[i]      |
| i 逐个递增            |
+----------------------+
        |
        v
输出 sum / mean / prod


优化后：分块归约、局部累积、块间合并

输入数组 A[0..N)
        |
        v
+------------------------------+
| 切分为 Block 0..k             |
+------------------------------+
        |
        v
+------------------------------+
| 每个 Block 内批量加载并累积   |
+------------------------------+
        |
        v
+------------------------------+
| 得到多个局部结果 partial sum |
+------------------------------+
        |
        v
+------------------------------+
| 块间合并为最终结果           |
+------------------------------+
        |
        v
输出 sum / mean / prod
```

**原理解释：**

普通逐元素累加会形成很长的依赖链：

```text
acc0 -> acc1 -> acc2 -> acc3 -> ...
```

每一步都依赖上一步结果，CPU 很难充分并行执行。分块归约的核心是把一个长依赖链拆成多个短依赖链：

```text
Block 0 -> partial sum 0
Block 1 -> partial sum 1
Block 2 -> partial sum 2
...
最后再合并 partial sum
```

这样可以减少单条依赖链长度，提高指令并行度和向量化效率。对于连续内存，批量加载还能更好利用 cache 和预取能力。

不过，浮点加法不是完全结合律，改变累加顺序可能带来极小数值差异。因此实现中需要控制误差范围，并对 `float32`、`float64`、整数、布尔、复数等不同 dtype 采用不同校验标准。对于无法保证数值一致性或误差可接受性的组合，应回退原生路径。

**优化前后效果对比：**

| 对比项     | 优化前                              | 优化后                      |
| ------- | -------------------------------- | ------------------------ |
| 处理方式    | 以原生通用归约路径为主，逐元素累计开销较明显           | 连续数据场景下采用批量加载、局部归约和块间合并  |
| 循环/分支开销 | 长数组场景循环次数多，标量依赖链较长               | 分块处理缩短依赖链，减少循环控制成本       |
| 访存/写回模式 | 大数组遍历依赖通用访存路径                    | 顺序访问连续数据块，提升缓存友好性        |
| 适用场景    | 通用 `sum`、`mean`，小规模或复杂 stride 场景 | 大数组、连续内存、规则 axis 的归约统计场景 |
| 预期效果    | `mean`、`sum` 等大数组统计吞吐受限          | 基础归约类统计路径吞吐提升，并支撑上层统计计算  |

---

### 2.2 最值与位置统计优化实现

该部分实现的功能，是提升 `max`、`argmax`、`nanargmax`、`nanargmin` 等扫描比较类统计路径的执行效率。

具体实现是对连续数据块进行批量扫描，在处理过程中同时维护当前局部最值和对应位置，在每一轮块处理完成后再将局部结果合并为全局结果。为了保持与 NumPy 原生行为一致，NaN 处理、边界值比较、相等值处理和 dtype 行为都必须严格对齐原生定义。

在 `argmax` 中，由于除了值本身还需要维护位置，因此实现时采用“值 + 索引”联合维护策略，块内完成批量比较后同步更新局部索引，最终再在块间完成归并。对不适合优化的位置统计场景，如极小数组或复杂 stride，直接回退原生路径。

**落地示意：**

```text
优化前：逐元素比较

输入数组 A[0..N)
        |
        v
+----------------------+
| 当前最值 cur          |
| 逐个比较 A[i]         |
| 必要时更新 cur / idx  |
+----------------------+
        |
        v
输出 min/max 和 index


优化后：块内批量比较、块间归并

输入数组 A[0..N)
        |
        v
+------------------------------+
| 切分为多个连续 Block         |
+------------------------------+
        |
        v
+------------------------------+
| 每个 Block 内批量比较         |
| 得到局部最值和局部 index      |
+------------------------------+
        |
        v
+------------------------------+
| 块间合并局部结果             |
+------------------------------+
        |
        v
输出全局 min/max 和 index
```

**原理解释：**

最值统计本质是扫描比较。逐元素实现每次只比较一个值：

```text
if A[i] > current_max:
    current_max = A[i]
    current_index = i
```

大数组下这个过程重复很多次。批量比较可以一次处理多个元素，先在块内找到局部最大值，再把多个局部最大值合并为全局最大值。

位置统计比单纯 `max` 更复杂，因为它还需要返回索引。优化路径必须保证：

* 最大值正确。
* 最大值对应的索引正确。
* 多个相同最大值时，返回位置与 NumPy 原生规则一致。
* NaN 相关行为与 NumPy 原生规则一致。

因此，`argmax` 优化不能只使用“批量求最大值”，还必须同时维护位置，并在块间归并时处理相等值和 NaN 规则。

**优化前后效果对比：**

| 对比项     | 优化前                                              | 优化后                       |
| ------- | ------------------------------------------------ | ------------------------- |
| 处理方式    | 逐元素扫描和比较为主                                       | 连续数据块中采用批量比较，并维护局部最值      |
| 循环/分支开销 | 长数组扫描中比较次数和分支判断开销明显                              | 批量比较减少逐元素分支，块间统一归并        |
| 访存/写回模式 | 顺序扫描但局部结果维护较分散                                   | 块内集中维护局部值和索引，最后合并写回结果     |
| 适用场景    | 通用 `max/argmax/nanargmax/nanargmin`，复杂 stride 场景 | 大数组、连续内存、规则扫描类最值统计场景      |
| 预期效果    | 大数组最值和位置统计扫描成本较高                                 | 扫描比较类统计路径吞吐提升，位置统计场景收益更稳定 |

---

### 2.3 方差与标准差优化实现

该部分实现的功能，是在保证数值可靠性的前提下，提高 `var`、`std` 路径的执行效率，支撑离散程度统计的热点计算场景。

具体做法是，在均值计算阶段优先复用已优化的归约路径，在偏差平方和累计阶段增加批量乘加处理；对适合分两阶段执行的场景，先完成均值计算，再完成偏差平方和批量归约；对部分连续数组和规则 dtype 场景，可在验证通过后采用受控的融合式遍历策略以减少访存次数。

实现中需要明确区分 `float32` 和 `float64` 的误差控制策略，确保与原生实现相比不存在不可接受的数值偏差；当某些路径存在精度风险时，应优先选择原生实现而非强行启用优化。

**落地示意：**

```text
优化前：原生通用多阶段计算

输入数组
   |
   v
+----------------------+
| 计算 mean             |
+----------------------+
   |
   v
+----------------------+
| 再次遍历计算偏差平方和 |
+----------------------+
   |
   v
+----------------------+
| 计算 var / std        |
+----------------------+


优化后：复用归约优化，并增强偏差平方和阶段

输入数组
   |
   v
+------------------------------+
| 复用批量归约路径计算 mean     |
+------------------------------+
   |
   v
+------------------------------+
| 分块批量计算 (x - mean)^2     |
| 并做局部累计                  |
+------------------------------+
   |
   v
+------------------------------+
| 合并局部平方和                |
+------------------------------+
   |
   v
+------------------------------+
| 计算 var / std                |
+------------------------------+
```

**原理解释：**

`var/std` 比 `mean` 更难优化，原因是它们不仅要累加，还要计算每个元素与均值的差，再平方，再累加。这个过程既有更多算术操作，也有更高数值风险。

常见风险包括：

* 大数相减可能导致精度损失。
* `float32` 的误差比 `float64` 更敏感。
* 改变累计顺序可能导致结果轻微变化。
* 复数、布尔、整数输入的 dtype 规则需要与 NumPy 保持一致。

因此，本 SR 对 `var/std` 的策略是保守增强：优先复用已经验证过的归约优化，再对偏差平方和阶段做批量化。只有在误差验证通过的 dtype 和 shape 组合上启用增强路径。

**优化前后效果对比：**

| 对比项     | 优化前                           | 优化后                      |
| ------- | ----------------------------- | ------------------------ |
| 处理方式    | 均值、偏差平方和等阶段依赖原生通用实现           | 复用归约优化，并增强偏差平方和累计阶段      |
| 循环/分支开销 | 多阶段计算带来重复遍历和标量累计开销            | 连续规则场景下减少部分重复循环和标量累计成本   |
| 访存/写回模式 | 多轮遍历可能产生较高访存成本                | 通过分块累计和受控融合减少部分重复访存      |
| 适用场景    | 通用 `var/std`，复杂 dtype 或精度敏感场景 | 大数组、连续内存、误差可控的离散程度统计场景   |
| 预期效果    | `var/std` 在大数组场景中受多阶段遍历影响明显   | 在保证数值可靠性的前提下提升离散程度统计路径性能 |

---

### 2.4 分位数与 NaN 分位数基础路径优化实现

该部分实现的功能，是提升 `percentile`、`nanpercentile` 等路径中基础步骤的执行效率，改善连续数据场景下的整体处理表现。

具体做法不是整体替换其原有选择或排序算法，而是增强其底层基础步骤。实现上优先对连续数组预处理、NaN 过滤、数据块复制、局部比较和中间缓冲搬运等步骤做优化，在可以保证语义不变的前提下，对这些规则性较强的部分采用批量化处理；对复杂的选择逻辑、非规则 stride 场景和高风险算法分支，则保留 NumPy 原生实现。

这一实现策略的目标不是让所有 `percentile/nanpercentile` 场景都强行命中优化路径，而是确保典型场景能够在基础步骤上获得收益，同时保持算法风险可控。

**落地示意：**

```text
优化前：整体依赖原生排序/选择/搬运路径

输入数组
   |
   v
+----------------------+
| 数据准备 / 复制       |
+----------------------+
   |
   v
+----------------------+
| 排序 / 选择 / 分区    |
+----------------------+
   |
   v
+----------------------+
| 计算中位数 / 分位数   |
+----------------------+


优化后：只增强规则基础步骤，核心算法保持原生语义

输入数组
   |
   v
+------------------------------+
| 连续数据块预处理增强          |
| 数据复制 / 缓冲搬运优化       |
+------------------------------+
   |
   v
+------------------------------+
| 可安全批量化的局部比较增强    |
+------------------------------+
   |
   v
+------------------------------+
| 复杂排序 / 选择逻辑保留原生   |
+------------------------------+
   |
   v
+------------------------------+
| 计算中位数 / 分位数           |
+------------------------------+
```

**原理解释：**

分位数计算依赖排序或选择。它不像 `sum` 那样只需要一次线性扫描。对于不同插值方法、不同分位点、不同 NaN 比例，内部路径可能不同。如果直接替换算法，风险较高。

因此，本 SR 只优化更底层、更稳定的基础步骤：

* 连续数据复制更快。
* NaN 数据过滤更快。
* 中间 buffer 搬运更快。
* 局部比较或预处理更快。

核心排序、选择、插值规则仍保持原生实现。这样做的好处是：即使性能收益不如完全重写算法大，但正确性风险明显更低，更适合作为平台增强型 SR。

**优化前后效果对比：**

| 对比项     | 优化前                                    | 优化后                     |
| ------- | -------------------------------------- | ----------------------- |
| 处理方式    | 整体依赖原生排序、选择和数据搬运路径                     | 不替换核心算法，重点增强基础预处理和搬运步骤  |
| 循环/分支开销 | 数据准备、局部比较、NaN 过滤和搬运过程中存在重复循环开销         | 对规则基础步骤进行批量化处理，减少部分循环成本 |
| 访存/写回模式 | 连续数据块复制和中间缓冲处理成本较高                     | 优化数据块复制、中间缓冲搬运和局部访问模式   |
| 适用场景    | 通用 `percentile/nanpercentile`，复杂选择逻辑场景 | 连续数组、规则数据块、可安全优化的预处理场景  |
| 预期效果    | 分位数在数据准备、NaN 过滤和搬运阶段存在开销               | 在语义不变前提下改善典型连续数据场景的处理效率 |

---

### 2.5 Histogram 优化实现

该部分实现的功能，是提升 `histogram`、`histogram2d` 等分桶统计路径的执行效率，改善桶索引计算和分桶累计过程中的热点开销。

对连续数据块，先通过批量比较和区间映射计算桶索引，再在局部缓冲区中进行分块计数，最后将各块结果合并写回全局桶数组；对于 `histogram2d`，则分别完成两个维度的索引映射，再进行二维桶定位和局部统计。

该实现的核心是减少随机写、提升局部性，并在可行场景下提升桶索引计算和边界判断效率。对桶数过大、内存布局复杂或局部缓存收益不明显的情况，可保留原生实现。

**落地示意：**

```text
优化前：逐元素计算桶并直接更新全局桶

输入数据 x[i]
     |
     v
+----------------------+
| 逐元素计算 bucket id  |
+----------------------+
     |
     v
+----------------------+
| 直接更新全局 bucket   |
+----------------------+
     |
     v
输出 histogram / bincount


优化后：批量算桶、局部统计、合并写回

输入数据块 Block
     |
     v
+------------------------------+
| 批量计算 bucket id            |
+------------------------------+
     |
     v
+------------------------------+
| 在局部桶缓存中分块统计        |
+------------------------------+
     |
     v
+------------------------------+
| 将局部桶结果合并到全局 bucket |
+------------------------------+
     |
     v
输出 histogram / bincount
```

**原理解释：**

直方图性能瓶颈通常不只是“算 bucket id”，还包括“更新 bucket”。如果每个元素都直接更新全局 bucket，可能导致大量随机写。尤其当很多元素集中落到少数 bucket 时，会频繁更新同一片内存，影响 cache 效率。

局部桶缓存的思想是：先在小范围局部 buffer 中统计一个数据块的结果，然后一次性合并到全局 bucket。这样可以减少全局随机写次数，提高局部性。

对于 `histogram2d`，还需要分别计算两个维度的 bucket id，再组合成二维桶位置，因此映射和边界判断更复杂。优化时需要保证边界规则与 NumPy 原生实现一致，例如：

* 落在范围外的数据是否忽略。
* 右边界是否包含。
* NaN/Inf 如何处理。
* bin 数量和边界数组如何解释。

**优化前后效果对比：**

| 对比项     | 优化前                      | 优化后                      |
| ------- | ------------------------ | ------------------------ |
| 处理方式    | 逐元素计算桶索引并直接更新桶结果         | 批量计算桶索引，局部统计后合并写回        |
| 循环/分支开销 | 边界判断、桶定位和计数更新重复发生        | 批量处理边界判断和索引映射，减少逐元素控制开销  |
| 访存/写回模式 | 直接更新全局桶，可能存在随机写和热点更新     | 使用局部桶缓存，分块统计后合并，降低热点写入成本 |
| 适用场景    | 通用 histogram，桶数过大或布局复杂场景 | 连续输入、桶规模适中、局部缓存有效的分桶统计场景 |
| 预期效果    | 分桶统计受索引计算和写回模式影响较大       | 分桶统计路径局部性提升，热点写入成本降低     |

---

### 2.6 综合统计链路、运行时选路、原生回退与验证实现

该部分实现的功能，是确保 statistics 优化只在收益明确且风险可控的场景下启用，并在不适配或异常时平滑回退到原生实现，同时完成数值可靠性和性能收益验证。

系统在运行时结合当前统计算子类型、dtype、ndim、axis 形式、shape、大小、内存连续性、stride 复杂度以及环境变量开关决定实际执行路径。对适配的主路径优先进入增强实现；当出现 dtype/axis/stride 组合未适配、数据规模过小、优化路径运行异常、结果校验失败、环境变量显式关闭优化或某组合已被标记为不稳定等情况时，必须自动回退至 NumPy 原生实现。

验证方面，要求覆盖不同 dtype、不同 axis、不同 shape、NaN/Inf 边界值和极端数据分布等情况，确保 `mean`、`sum`、`var`、`std`、`max`、`argmax`、`nanargmax`、`nanargmin`、`percentile`、`nanpercentile`、`histogram` 等路径与原生实现保持一致；同时结合官方 benchmark 评估性能收益、回退比例和整体链路表现。

**落地示意：**

```text
统计运算请求
   |
   v
+-------------------------------+
| 读取 op / dtype / axis / shape |
| stride / size / 内存连续性     |
+-------------------------------+
   |
   v
+-------------------------------+
| 判断是否属于主优化场景         |
+-------------------------------+
   |
   +---- 不适配 / 小数据 / 复杂布局 ----> NumPy 原生路径
   |
  适配
   |
   v
+-------------------------------+
| 进入统计增强路径               |
+-------------------------------+
   |
   +---- 异常 / 校验失败 ----> NumPy 原生路径
   |
  成功
   |
   v
+-------------------------------+
| 输出结果并记录验证与性能信息   |
+-------------------------------+
```

**原理解释：**

统计优化必须非常重视回退机制，原因有三点。

第一，统计函数的输入组合非常多。不同 dtype、axis、shape、stride、NaN 比例和内存布局，可能触发完全不同的内部路径。一个优化在连续大数组上有效，不代表在复杂 stride 或小数组上也有效。

第二，统计结果对数值误差敏感。尤其是 `var/std`、`percentile`、`nanpercentile` 这类路径，如果处理顺序或边界逻辑稍有变化，就可能导致结果差异。

第三，小数据量场景可能出现性能倒挂。优化路径通常需要额外判断、分块、初始化局部 buffer。如果输入很小，这些额外开销可能超过收益。

因此，本 SR 采用如下策略：

```text
规则主场景：
大数组 + 连续内存 + 常见 dtype + 规则 axis
        -> 启用增强路径

风险场景：
小数组 + 复杂 stride + 特殊 dtype + 精度敏感 + 未验证组合
        -> 回退 NumPy 原生路径
```

同时记录路径信息，便于 SE 判断某个 case 是否命中优化路径。如果 benchmark 未达到预期收益，需要先确认是否回退，再分析是阈值问题、dtype 未命中、axis 不适配，还是优化路径本身收益不足。

**优化前后效果对比：**

| 对比项     | 优化前                        | 优化后                                            |
| ------- | -------------------------- | ---------------------------------------------- |
| 处理方式    | 主要依赖原生统计实现，缺少平台感知策略        | 根据 op、dtype、axis、shape、stride、规模等动态选择增强路径或原生路径 |
| 循环/分支开销 | 不区分规则场景和复杂场景，无法规避不必要的增强开销  | 通过阈值、白名单和回退策略控制增强路径启用范围                        |
| 访存/写回模式 | 不针对内存连续性和 stride 复杂度做差异化处理 | 连续规则场景走增强路径，复杂布局保留原生路径                         |
| 适用场景    | 所有场景统一依赖原生路径               | 主路径增强，复杂 axis、复杂 stride、小数据量和异常场景回退            |
| 预期效果    | 性能收益和异常原因难以体系化归档           | 形成“主路径加速、复杂场景稳定、结果可验证”的闭环                      |

---

# 三、RNG 与 Statistics SR 的整体关系

RNG 和 Statistics 虽然是两个 SR，但在实际业务链路中经常连续出现：

```text
RNG 生成随机数据
        |
        v
Statistics 做均值、方差、分位数、直方图等分析
```

因此，本次设计中两者的关系可以概括为：

| SR            | 主要优化对象                                           | 核心收益来源               | 风险控制重点                |
| ------------- | ------------------------------------------------ | -------------------- | --------------------- |
| RNG SR        | 随机 bit、整数随机数、浮点随机数、normal 底层路径                   | 批量生成、批量转换、连续写回       | 不改变算法语义、seed 行为和随机数质量 |
| Statistics SR | mean、sum、var、std、max、argmax、percentile、histogram | 批量加载、局部累计、块间合并、局部桶缓存 | 不改变统计定义、边界行为和数值可靠性    |

整体设计原则是：

```text
主路径增强
复杂场景回退
结果正确优先
性能收益可验证
问题定位可追踪
```

这样可以在鲲鹏 920B 平台上针对高频大数组场景形成稳定收益，同时避免因为过度优化导致接口行为、数值结果或随机数质量出现不可控风险。
