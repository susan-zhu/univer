## 1. 验证方法比较

### 1.1 自回归 Baseline

目标模型每轮生成一个 token，不构造草稿树，也不执行接受/拒绝采样。它是衡量端到端加速比的基础。

优点是实现简单、额外开销最小，并且不存在草稿质量问题。缺点是每次前向只能得到一个新 token，无法摊薄目标模型调用成本。

### 1.2 Vanilla RRSw

RRSw（Recursive Rejection Sampling without replacement）从已接受的父节点出发，按顺序验证其子节点。若候选被拒绝，则从目标概率中扣除 draft 概率，同时从 draft 分布中移除已拒绝 token 并重新归一化，然后继续验证下一个兄弟节点；接受后递归进入下一层。

RRSw 的优势是逻辑直接、无放回采样避免重复候选，并且在 EAGLE 中已有相对成熟的实现。局限是兄弟候选按顺序处理，每次拒绝后都需要更新残差分布，横向候选之间缺少联合最优分配，且存在串行依赖。

### 1.3 Traversal Verification

Traversal Verification 将 Block Verification 扩展到树结构，使用后序遍历从底向上验证节点。它利用整条前缀的联合接受概率优化纵向决策；当叶节点被拒绝时，算法回退到兄弟节点或父节点。横向候选仍采用 RRSw 风格的局部拒绝采样。

它通常能获得略高于 Vanilla RRSw 的接受长度，但实现成本更高：

- 后序决策是动态、串行的，难以完全向量化。
- 每个被访问节点都可能触发 GPU 标量判断与 CPU-GPU 同步。
- 每次拒绝都会改变父节点的 target/draft residual distribution，并影响后续子树的接受率。
- 如果直接实现，拒绝后可能扫描活动树并重算大量后代节点。

因此，Traversal 的理论接受长度优势不一定能够转化为端到端吞吐优势。

### 1.4 Greedy 方法

Greedy 方法属于单步 multi-draft 最优传输方法。它在每层选择 Top-(`m-1`) token，并从残差 draft 分布中采样最后一个 token，然后在同一个概率空间中联合分配多个候选的接受质量。

它能够优化同层兄弟候选，接受率通常高于 RRSw，但各层仍独立处理，没有充分利用跨深度的前缀依赖。当前复现代码尚未完成该方法的独立 benchmark，因此本报告只列出论文参考值，后续需要补测。

### 1.5 UniVer

UniVer 同时处理横向 multi-draft 和纵向 multi-step 依赖。分配阶段按深度并行计算同层节点，决策阶段只进行一次后序遍历。与 Traversal 相比，它将大量动态 residual 更新转换为可预计算、可向量化的概率分配，因此更容易获得实际吞吐收益。

其代价是需要特定的混合草稿采样方式、保存额外的节点概率和拓扑元数据，并执行目标模型全词表概率计算。实现时必须保证 tree sampling law 与 verifier 使用的 proposal probability 完全一致，否则会破坏无损性。

## 2. 结果对比
### 2.1 Vicuna模型运行命令

```bash
python -m eagle.evaluation.benchmark_verify_methods \
  --base-model-path /data/public/zhusuzhen/models/Vicuna-7B-v1.3 \
  --ea-model-path /data/public/zhusuzhen/models/EAGLE-Vicuna-7B-v1.3 \
  --output-dir /home/zhusuzhen/data/verify_vicuna \
  --question-file /home/zhusuzhen/data/question.jsonl \
  --verify-methods greedy univer  RRSw traversal_verification  \
  --chat-template vicuna \
  --no-use-eagle3 \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-new-tokens 256 \
  --total-token 63 \
  --depth 5 \
  --draft-top-k 2 \
  --warmup 3 \
  --seed 41 \
  --compile-univer \
  --traversal-backend  cpu_lazy \
  --traversal-pinned-buffer  \
  --turns all \
  --limit 10
```

### 2.2 Llama模型运行命令
```bash
python -m eagle.evaluation.benchmark_verify_methods \
  --base-model-path /data/public/zhusuzhen/models/Llama-3.1-8B-Instruct \
  --ea-model-path /home/zhusuzhen/data/models/EAGLE3-LLaMA3.1-Instruct-8B \
  --output-dir /home/zhusuzhen/data/verify_univer_llama \
  --question-file /home/zhusuzhen/data/question.jsonl \
  --verify-methods RRSw traversal_verification greedy univer \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-new-tokens 256 \
  --total-token 63 \
  --depth 5 \
  --draft-top-k 2 \
  --warmup 3 \
  --seed 45 \
  --compile-univer \
  --traversal-backend  cpu_lazy \
  --traversal-pinned-buffer  \
  --limit 10
```
`注意：`当设置`--limit 10`时，只执行question-file数据集里面的10条

`--verify-methods` 后面可以更一个或多个要验证的方法

当是 
` --total-token 63 \
  --depth 5 \
  --draft-top-k 2 
`
表示深度为5，宽度为2的二叉树，共计63个点，二叉树各层节点数分别是1,2,4,8，16,32，符合论文中5层32个叶子节点

