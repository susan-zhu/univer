
## 摘要

本文在 EAGLE3 框架中复现 UniVer 论文 Table 2 所涉及的验证方法，并针对草稿树构造、验证函数、全词表概率计算以及 CPU-GPU 同步进行了实现优化。目标模型为 Llama-3.1-8B-Instruct，草稿模型为 EAGLE3-LLaMA3.1-Instruct-8B，本地设备为单张 NVIDIA L20X 144 GB；论文使用 NVIDIA RTX A6000 48 GB。

当前 10 条样本预实验表明，优化后 RRSw、Traversal Verification 和 UniVer 的吞吐分别达到 53.98、42.10 和 65.09 tokens/s。其中 UniVer 的平均接受长度为 2.57，已经接近论文 Table 2 中 translation 域的 2.57；但 RRSw 和 Traversal 的平均接受长度仍低于论文对应结果。Traversal 的接受长度高于 RRSw，但验证轮次延迟明显更大，说明其串行后序遍历、拒绝后的残差概率更新及主机同步抵消了接受长度收益。

需要强调的是，当前结果来自 `--limit 10`。本地 `question.jsonl` 含 480 条 Spec-Bench 数据，而前 10 条全部属于 translation 域，因此该实验属于功能与性能预实验，不是对 Table 2 六个领域平均结果的完整复现。严格复现还需完成 480 条数据、3 个随机种子、统一 baseline 参数及 Greedy 方法实验。

## 1. 前言

### 1.1 背景介绍

大语言模型的自回归推理一次通常只产生一个 token，每个 token 都需要执行一次目标模型前向计算，因此生成阶段容易受到内存带宽、KV Cache 访问和 kernel launch 开销的限制。推测解码通过“草稿模型生成候选、目标模型并行验证候选”的方式，在保持目标模型输出分布不变的前提下，使一次目标模型调用能够确认多个 token。

树形推测解码同时包含两个优化维度：

- **Multi-draft（横向）**：同一深度提供多个候选 token，提高至少接受一个候选的概率。
- **Multi-step（纵向）**：沿候选路径连续生成多步，并联合考虑前缀的接受概率，提高每轮可确认的 token 数。

以往方法通常只重点优化其中一个维度。例如，RRSw 在每个节点逐个验证兄弟候选，Traversal Verification 通过后序遍历增强纵向联合验证，但横向仍使用 RRSw；Greedy 方法使用最优传输思想优化单层多个候选，但没有完整建模跨层依赖。

### 1.2 UniVer 方法概述

UniVer 将树形验证拆分为两个阶段：

1. **自顶向下的概率分配阶段**：将父节点的有效接受概率作为动态缩放因子，在同一层的兄弟节点之间求解条件最优传输问题，并把有效概率质量继续传播到下一层。
2. **后序决策阶段**：按照后序顺序对节点执行一次接受判定；一旦某节点被接受，立即终止遍历，并输出对应路径。

UniVer 使用 Top-(`m-1`) 加一次 residual sampling 的混合草稿策略。对于二叉树，即每个展开父节点保留一个 draft 概率最高的 token，再从移除该 token 后的残差分布中采样另一个 token。该采样结构使兄弟节点的接受概率具有闭式表达，不需要在运行时调用通用最优传输求解器。

论文从理论上给出三个主要性质：

- **无损性**：最终采样分布保持为目标模型分布。
- **条件最优性**：给定父节点有效接受概率和指定草稿策略时，单层子树达到最优接受率。
- **相对 Greedy 的非劣性**：在论文条件下，UniVer 的期望接受长度不低于单步 Greedy 方法。

## 2. 验证方法比较

### 2.1 自回归 Baseline

目标模型每轮生成一个 token，不构造草稿树，也不执行接受/拒绝采样。它是衡量端到端加速比的基础。

优点是实现简单、额外开销最小，并且不存在草稿质量问题。缺点是每次前向只能得到一个新 token，无法摊薄目标模型调用成本。

### 2.2 Vanilla RRSw

RRSw（Recursive Rejection Sampling without replacement）从已接受的父节点出发，按顺序验证其子节点。若候选被拒绝，则从目标概率中扣除 draft 概率，同时从 draft 分布中移除已拒绝 token 并重新归一化，然后继续验证下一个兄弟节点；接受后递归进入下一层。

RRSw 的优势是逻辑直接、无放回采样避免重复候选，并且在 EAGLE 中已有相对成熟的实现。局限是兄弟候选按顺序处理，每次拒绝后都需要更新残差分布，横向候选之间缺少联合最优分配，且存在串行依赖。

### 2.3 Traversal Verification

Traversal Verification 将 Block Verification 扩展到树结构，使用后序遍历从底向上验证节点。它利用整条前缀的联合接受概率优化纵向决策；当叶节点被拒绝时，算法回退到兄弟节点或父节点。横向候选仍采用 RRSw 风格的局部拒绝采样。

它通常能获得略高于 Vanilla RRSw 的接受长度，但实现成本更高：

- 后序决策是动态、串行的，难以完全向量化。
- 每个被访问节点都可能触发 GPU 标量判断与 CPU-GPU 同步。
- 每次拒绝都会改变父节点的 target/draft residual distribution，并影响后续子树的接受率。
- 如果直接实现，拒绝后可能扫描活动树并重算大量后代节点。

因此，Traversal 的理论接受长度优势不一定能够转化为端到端吞吐优势。

### 2.4 Greedy 方法

Greedy 方法属于单步 multi-draft 最优传输方法。它在每层选择 Top-(`m-1`) token，并从残差 draft 分布中采样最后一个 token，然后在同一个概率空间中联合分配多个候选的接受质量。

它能够优化同层兄弟候选，接受率通常高于 RRSw，但各层仍独立处理，没有充分利用跨深度的前缀依赖。当前复现代码尚未完成该方法的独立 benchmark，因此本报告只列出论文参考值，后续需要补测。

### 2.5 UniVer

UniVer 同时处理横向 multi-draft 和纵向 multi-step 依赖。分配阶段按深度并行计算同层节点，决策阶段只进行一次后序遍历。与 Traversal 相比，它将大量动态 residual 更新转换为可预计算、可向量化的概率分配，因此更容易获得实际吞吐收益。

其代价是需要特定的混合草稿采样方式、保存额外的节点概率和拓扑元数据，并执行目标模型全词表概率计算。实现时必须保证 tree sampling law 与 verifier 使用的 proposal probability 完全一致，否则会破坏无损性。

### 2.6 综合比较

| 方法 | 横向 multi-draft | 纵向 multi-step | 主要执行方式 | 优点 | 主要局限 |
|---|---|---|---|---|---|
| 自回归 | 否 | 否 | 单 token 生成 | 简单、额外开销低 | 每次前向只生成一个 token |
| RRSw | 局部顺序处理 | 逐层递归 | 无放回拒绝采样 | 实现成熟、候选不重复 | 串行验证，横向非全局最优 |
| Traversal | RRSw 局部处理 | 联合优化 | 动态后序遍历 | 接受长度通常高于 RRSw | 拒绝后残差更新和同步开销大 |
| Greedy | 单层 OT 优化 | 各层独立 | Top-(`m-1`) + residual sample | 单层多候选接受率高 | 未充分利用跨层依赖 |
| UniVer | 条件 OT 联合分配 | 概率质量跨层传播 | 自顶向下分配 + 一次后序决策 | 接受长度高、同层易并行 | 依赖特定采样结构和较复杂元数据 |

## 3. 实验规划与配置

### 3.1 复现目标

本项目以论文 Table 2 为主要目标，验证以下结论：

1. 在 Llama-3.1-8B-Instruct 上，UniVer 的平均接受长度高于 RRSw、Traversal 和 Greedy。
2. 接受长度提升能够转化为端到端 tokens/s 提升。
3. 在温度为 1.0 的随机采样条件下，复现各方法的相对排序。
4. 分离算法收益和工程实现开销，确定草稿树构造、概率计算、验证控制流和设备同步分别占用的时间。

论文 Table 2 的参考结果如下：

| 方法 | MT | Trans. | Summ. | QA | Math | RAG | 平均接受长度 τ | 平均 TPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RRSw | 2.96 | 2.31 | 2.56 | 2.33 | 3.31 | 2.86 | 2.72 ± 0.02 | 48.5 ± 0.2 |
| Traversal | 3.05 | 2.34 | 2.60 | 2.40 | 3.37 | 2.94 | 2.78 ± 0.02 | 49.5 ± 0.4 |
| Greedy | 3.09 | 2.54 | 2.70 | 2.46 | 3.45 | 3.04 | 2.88 ± 0.02 | 50.6 ± 0.4 |
| UniVer | **3.19** | **2.57** | **2.77** | **2.57** | **3.51** | **3.11** | **2.95 ± 0.01** | **52.2 ± 0.1** |

论文的 Llama-3.1-8B-Instruct 自回归 baseline 为 34.5 tokens/s。对应端到端加速比分别约为 1.41×、1.43×、1.47× 和 1.51×。

### 3.2 论文配置与本地配置

| 配置项 | 论文 | 当前本地实验 |
|---|---|---|
| 目标模型 | Llama-3.1-8B-Instruct | Llama-3.1-8B-Instruct |
| 草稿模型 | EAGLE | EAGLE3-LLaMA3.1-Instruct-8B |
| 数据集 | Spec-Bench，6 个领域 × 80 条 | `question.jsonl` 共 480 条；当前只运行前 10 条 translation |
| GPU | NVIDIA RTX A6000 48 GB | NVIDIA L20X 144 GB；服务器共 4 卡，实验固定单卡 0 |
| 树结构 | 平衡二叉树，depth=5，63 个节点 | `total-token=63`、`depth=5`、`draft-top-k=2` |
| temperature | 1.0 | 1.0 |
| top-p | 论文未进行 top-p 截断 | 1.0 |
| target top-k | 未截断 | 0（验证命令默认值） |
| 最大生成长度 | 未在 Table 2 附近明确给出 | 256 tokens |
| 随机种子 | 3 个随机种子，报告 mean ± std | 当前单 seed，默认 42 |
| warmup | 未明确报告 | 每种方法 1 次 |
| 精度 | 未明确报告 | FP16 |

`total-token=63` 传入 EAGLE 模型后，内部将 draft token 预算设为 62，再加上目标模型预采样的 root/bonus token，形成 63 节点树。

### 3.3 当前运行命令

```bash
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.benchmark_verify_methods \
  --base-model-path /data/public/zhusuzhen/models/Llama-3.1-8B-Instruct \
  --ea-model-path /home/zhusuzhen/data/models/EAGLE3-LLaMA3.1-Instruct-8B \
  --question-file /home/zhusuzhen/data/question.jsonl \
  --verify-methods RRSw traversal_verification univer \
  --output-dir /home/zhusuzhen/data/verify_univer \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-new-tokens 256 \
  --total-token 63 \
  --depth 5 \
  --draft-top-k 2 \
  --limit 10
```

### 3.4 评估指标

#### 平均接受长度

本地代码采用论文“每个验证周期产生的平均 token 数”定义：

```text
average_accept_length = total_accept_length / verification_rounds
```

每轮接受长度包含接受的 draft token，以及该轮必然输出的一个 root/bonus token。因此：

```text
average_accept_length = average_accepted_draft_tokens + 1
```

该指标主要衡量验证算法的理论效率，不直接包含验证函数本身的执行成本。

#### 吞吐

```text
tokens_per_second = output_tokens / measured_generation_seconds
```

计时前后均调用 `torch.cuda.synchronize()`，包含生成阶段的草稿树构造、目标模型 tree forward、验证、KV Cache 更新及 Python 控制流，不包含模型加载和 warmup。

#### 每轮耗时

```text
milliseconds_per_round = seconds / verification_rounds × 1000
```

该指标可以将接受长度和实现开销分开，是分析 Traversal 性能问题的重要指标。

#### 加速比

```text
speedup = speculative_tokens_per_second / baseline_tokens_per_second
```

baseline 必须与推测方法使用相同的模型实现、设备、精度、数据、temperature、top-p、target top-k 和计时范围。

### 3.5 后续严格复现实验计划

1. 在同一个 `benchmark_verify_methods.py` 进程中同时测试 baseline、RRSw、Traversal 和 UniVer，避免模型加载与计时路径差异。
2. 使用 `--limit 0` 跑完整 480 条 Spec-Bench，并按论文六个领域聚合结果。
3. 分别使用 seed 42、43、44 运行，报告 mean ± std。
4. 完成 Greedy verifier 后加入同一实验矩阵。
5. 明确 MT-Bench 两轮对话处理方式，并与 Spec-Bench 官方协议对齐；如果使用 `--turns all`，需要避免 MT 域因两轮记录而在总体平均中被重复加权。
6. 每个方法至少 warmup 3 次，记录 CUDA、PyTorch、Transformers、驱动版本和 GPU 时钟状态。

推荐的单 seed 对齐命令为：

```bash
CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.benchmark_verify_methods \
  --base-model-path /data/public/zhusuzhen/models/Llama-3.1-8B-Instruct \
  --ea-model-path /home/zhusuzhen/data/models/EAGLE3-LLaMA3.1-Instruct-8B \
  --question-file /home/zhusuzhen/data/question.jsonl \
  --verify-methods baseline RRSw traversal_verification univer \
  --output-dir /home/zhusuzhen/data/verify_univer_seed42 \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 0 \
  --max-new-tokens 256 \
  --total-token 63 \
  --depth 5 \
  --draft-top-k 2 \
  --limit 0 \
  --turns first \
  --warmup 3 \
  --seed 42
```


