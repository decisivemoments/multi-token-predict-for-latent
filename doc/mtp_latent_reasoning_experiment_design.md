# MTP 与 Sentence-level Latent Reasoning：当前研究设计

本文档是当前主设计文档。历史实验设想和 transition-first 方案放在 `doc/archive/`。

## 0. 总体假设与研究分叉

本项目最初的总假设是：

```text
MTP 比 NTP 更容易学到长程语义；长程语义有利于改善 latent reasoning。
```

这里的“长程语义”在当前项目中先操作性定义为： sentence-level representation

这个总假设可以拆成两个子假设：

```text
子假设 1: MTP 比 NTP 更能帮助模型学到 sentence-level predictive representation。 子假设 2: 这种 sentence-level predictive representation 有利于 latent reasoning。
```

### 子假设 1 的当前状态

```text
实验: encoder-decoder codec
输入: question + s_1 + ... + s_i
目标: 预测 s_{i+1} 或 answer
```

当前实验现象说明：

```text
MTP init 的模型经过 codec 训练后，在通过 prefix 预测后续 step 的任务上表现更准确； 因此，子假设 1 已经得到初步支持。
这只是从实验结果上支持“MTP 有助于学到 sentence-level 表征”。 它还没有解释为什么 MTP 会这样。 它也没有证明这种表征一定能改善 latent reasoning。
```

### 当前分叉

在子假设 1 得到初步支持后，有两个自然方向：

```text
方向 A: 深挖子假设 1
    为什么 MTP 更容易学到 sentence-level predictive representation？
    MTP 学到的表征和 NTP 学到的表征具体差在哪里？

方向 B: 检验子假设 2
    已经学到的 sentence-level latent z，能否被用于 latent reasoning？
    它能否支持状态转化、递推、搜索、或其他 reasoning-time computation？
```






当前项目优先选择方向 B。

原因：

```text
如果 z 不能被用于 latent reasoning，
那么继续解释 MTP 为什么学到 z 的价值有限。

如果 z 能被用于 latent reasoning，
再回头分析 MTP 为什么更容易产生这种 z，才更有研究意义。
```

因此，当前主问题是：

```text
基于已经学到的 sentence-level predictive representation，
我们如何判断它是否具有 latent reasoning 所需的状态结构？
以及如何把它用于状态转化或 reasoning-time computation？
```

## 1. 当前研究问题

当前更准确的问题不是“继续证明 MTP 是否更好”，而是：

```text
encoder-decoder codec 已经学到了 decoder-readable、prefix-conditioned 的 sentence-level latent z。
z 具有什么性质时，神经网络才能抓取其中的状态特征，并用于状态转化或 latent reasoning？
```

MTP/NTP init 的差异暂时降级为背景变量。当前优先目标是找到 `z` 的可用状态结构，而不是继续围绕 MTP/NTP 做单一胜负比较。

## 2. 已验证结论

当前 encoder-decoder 架构已经能把 reasoning prefix 压成一个 decoder-readable latent：

```text
Encoder(prefix) -> z
Decoder(z) -> current step / answer
```

其中：

```text
prefix = question + previous_steps
target = current_step 或 final answer
```

本地 `output/` 记录显示，GSM Exp1 codec 在 epoch 25 附近已经达到：

```text
best valid loss: 0.4559
best valid token_h1_acc: 0.8784
best valid answer_acc: 0.8684
```

valid generation 样例显示：

```text
decoder 能从 latent 直接生成完整 step / answer
生成能正常以 EOS 结束
错误多表现为变量选择或算术操作错误，而不是格式完全崩溃
```

这说明当前 codec 至少建立了可读的、局部预测性的 sentence-level latent interface。

## 3. Representation Usefulness 诊断结论

### 3.1 Candidate Ranking

实现口径：

```text
prefix_i = question + s_1 + ... + s_{i-1}
gold_i = s_i 或 answer
z_i = Encoder(prefix_i)

candidates = gold_i + same-question negatives + cross-question negatives
score(z_i, candidate) = - mean teacher-forced token NLL under Decoder(z_i)
```

GSM Exp1 结果：

```text
MTP top1_accuracy ~= 0.855
MTP MRR ~= 0.913

NTP top1_accuracy ~= 0.867
NTP MRR ~= 0.919
```

解释：

```text
z 通过 decoder readout 可以作为 next-step candidate scorer。
MTP 与 NTP 在该口径下基本持平，NTP 略高。
因此当前重点不应是继续比较 MTP/NTP，而是把 scorer 用起来。
```

### 3.2 Latent-as-Verifier

实现口径：

```text
positive = (prefix_i, gold_i)
negative = (prefix_i, hard / same-question / cross-question non-gold candidate)
score = - mean teacher-forced token NLL under Decoder(z_i)
valid if score >= threshold
```

当前 hard negatives：

```text
hard_wrong_result
hard_wrong_operator
hard_wrong_operand
hard_wrong_answer
```

GSM Exp1 结果：

```text
MTP AUC ~= 0.972
NTP AUC ~= 0.971
```

解释：

```text
Decoder(z) 的条件似然可以作为 verifier signal。
MTP/NTP 基本持平。
NTP 更保守，precision 更高。
MTP 更宽松，recall 更高。
```

当前 hard negatives 还偏容易，很多是算式内部不一致。下一轮若继续做 verifier 诊断，应使用内部自洽但上下文错误的 hard negatives。

## 4. 当前 Claim Boundary

可以说：

```text
z 是 decoder-readable 的 sentence-level latent。
z 可以支持 step / answer generation。
z 可以作为 next-step candidate scorer。
z 可以作为初步 verifier signal。
```

不能说：

```text
z 已经是可递推 reasoning state。
z_i -> z_{i+1} 是自然或容易学习的。
continuous latent rollout 已经成立。
MTP 在当前 ranking / verifier 诊断中明显优于 NTP。
```

## 5. 当前先验层级

从已验证结论出发，当前应优先使用较弱但已得到支持的先验。

### A. Decoder-readable representation

已基本验证：

```text
z 能被 decoder 读出当前 step 或 answer。
```

可用方式：

```text
step generation
answer generation
EOS-based step boundary
```

### B. Predictive / discriminative representation

已得到初步支持：

```text
z 可以区分当前 prefix 下的 gold next step 和负例候选。
```

可用方式：

```text
candidate scoring
reranking
accept/reject verifier
search-time critic
```

### C. Trajectory / transition representation

尚未验证，但这是当前真正关心的问题：

```text
z 是否具有可被神经网络抓取的状态转化结构。
```

这里不能直接跳到 full rollout。应该先做局部、可证伪的 transition-readability probe。

## 6. 下一阶段：Transition Readability

当前下一阶段目标：

```text
判断已经学到的 sentence-level latent z 是否具有可被神经网络读取并用于状态转化的结构。
```

### 6.1 Falsifiable Conjecture

```text
如果 z 是 latent reasoning 可用的状态表征，
那么存在一个简单神经模块 P，
能够从当前状态 z_i 或历史状态 z_1...z_i 中预测下一状态 z_{i+1}
或检索出正确的下一状态。
```

这不是直接验证 continuous rollout，而是验证：

```text
z 中是否存在可被网络抓取的 transition-relevant feature。
```

### 6.2 Physical Priors

```text
Prior A: reasoning trace 有状态推进
    prefix_i = question + previous_steps
    prefix_{i+1} = prefix_i + s_i

Prior B: codec latent z_i 至少包含当前 prefix 对下一步的预测性信息
    6.1 / 6.2 ranking 和 verifier 已经支持这一点。

Prior C: 如果 z 是好的 latent reasoning state，
    z_i 应该携带足够信息，让简单模型预测或检索 z_{i+1}。
```

### 6.3 Mathematical Models

```text
z_i = Encoder(question + s_1 + ... + s_{i-1})
z_{i+1} = Encoder(question + s_1 + ... + s_i)

absolute transition:
    P(z_i) -> z_{i+1}

delta transition:
    P(z_i) -> Δ_i = z_{i+1} - z_i
    z_hat_{i+1} = z_i + P(z_i)

retrieval transition:
    score(P(z_i), z_candidate)
    rank gold z_{i+1} among negatives

history-conditioned transition:
    P(z_1, ..., z_i) -> z_{i+1}
```

### 6.4 Minimal Experiments

#### Probe 1: next-latent retrieval

```text
input:
    z_i

output:
    query vector q_i = P(z_i)

metric:
    rank gold z_{i+1} among same-batch / same-question / cross-question candidates

success:
    gold z_{i+1} rank 显著高于 random baseline

failure interpretation:
    如果 retrieval 失败，z_i 可能没有容易读取的 next-state signal。
```

#### Probe 2: delta prediction

```text
input:
    z_i

target:
    Δ_i = z_{i+1} - z_i

model:
    small MLP

metrics:
    cosine(z_i + predicted_delta, z_{i+1})
    MSE(z_i + predicted_delta, z_{i+1})
    retrieval rank of z_i + predicted_delta

failure interpretation:
    如果 retrieval 可行但 delta regression 失败，说明 next-state information 存在，但 latent geometry 不适合直接回归。
```

#### Probe 3: history-conditioned transition

```text
input:
    z_1, ..., z_i

target:
    z_{i+1}

model:
    small transformer / GRU / pooled MLP

metrics:
    same as retrieval / delta prediction

failure interpretation:
    如果 history-conditioned probe 明显优于 z_i-only，
    说明单个 z_i 不是 Markov state，latent reasoning 需要历史状态或 recurrent state。
```

#### Probe 4: decoder-readability of predicted latent

```text
input:
    predicted z_hat_{i+1}

test:
    Decoder(z_hat_{i+1}) 是否能生成 gold next target s_{i+1} / answer

metrics:
    token_acc
    exact match
    EOS finish rate

failure interpretation:
    如果 z_hat 接近 z_{i+1} 但 decoder 失败，
    说明 decoder-readable manifold 很窄，transition 输出必须额外约束。
```

## 7. 决策规则

```text
如果 retrieval / delta / history probes 都失败：
    当前 z 可能不是合适的 transition state。
    下一步应重新设计 codec objective，而不是调 transition model。

如果 retrieval 成功但 delta regression 失败：
    z 有 next-state information，但几何不适合绝对 latent 回归。
    可尝试 retrieval-based transition 或 contrastive objective。

如果 delta 单步成功但 decoder-readability 失败：
    问题在 decoder-readable manifold。
    需要 manifold regularization 或 decode-aware transition loss。

如果 z_i-only 失败但 history-conditioned 成功：
    当前 z 不是 Markov state。
    latent reasoning 需要 history model，而不是单状态 transition。

如果单步 probe 成功但 multi-step rollout 失败：
    问题在递推稳定性。
    再考虑 rollout-specific training。
```

## 8. 暂停项

下面这些继续暂停，直到 transition-readability probes 给出局部证据：

```text
continuous rollout
MTP transition init
大模型 transition backbone
端到端 transition rollout
```

## 9. 当前最小研究链条

按 research-exploration，这条链必须保持可证伪：

```text
子假设 1:
    MTP 有助于学到 sentence-level predictive representation。
    当前 codec 实验已初步支持。

子假设 2:
    sentence-level predictive representation 有利于 latent reasoning。
    当前尚未验证。

下一步:
    不是直接 rollout，
    而是测试 z 是否有可被简单神经模块读取的 transition structure。
```

这避免把“z 能生成 / 能打分”直接跳成“z 能递推”。如果 transition-readability probe 失败，失败本身就是有效研究结果：它说明当前 codec 表征虽然有长程预测信息，但未必是 latent reasoning state。
