# MTP 与 Sentence-level Latent Reasoning：当前研究设计

本文档是当前主设计文档。历史实验设想和已经降级的 transition 方案放在 `doc/archive/` 或对应专项文档中。

## 1. 当前研究问题

最初问题是：

```text
MTP 相比 NTP，是否能改善 latent reasoning？
```

当前更准确的问题是：

```text
MTP 作为 future-predictive pretraining signal，是否能帮助模型学出更有用的 sentence-level predictive representation？
如果这个 representation 已经学到，下一步应该怎样使用它来改善 latent state reasoning？
```

这两个问题不能混在一起。已经验证的是 codec 表征能力；尚未验证的是这些表征是否天然支持 transition、continuous rollout 或多步 latent reasoning。

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

从当前本地 `output/` 记录看，`exp1_gsm_ntp_epoch25` 已经达到：

```text
best valid loss: 0.4559
best valid token_h1_acc: 0.8784
best valid answer_acc: 0.8684
```

`output/epoch_025.json` 中的 valid generation 样例显示：

- decoder 能从 latent 直接生成完整 step 或 answer；
- 生成能正常以 EOS 结束；
- 错误多表现为变量选择或算术操作错误，而不是格式完全崩溃。

这说明当前 codec 至少学到了可读的、局部预测性的 sentence-level latent interface。

## 3. 当前 claim boundary

可以说：

```text
当前 codec latent z 对 decoder 是 readable 的，并且能支持 prefix-conditioned step / answer generation。
MTP init 在表征生成器方向上已有正信号。
```

不能说：

```text
z 已经是可递推 reasoning state。
z_i -> z_{i+1} 是自然或容易学习的。
transition model 只要设计好就能 rollout。
continuous latent rollout 已经成立。
MTP transition init 已经被证明更适合 latent dynamics。
```

之前直接设计 transition model 属于强假设跳步。它隐含了许多尚未验证的前提，例如 latent trajectory 的局部结构、decoder 对 off-manifold latent 的鲁棒性、以及 transition 输出仍在 decoder-readable manifold 上。

## 4. 当前先验层级

从已验证的 codec 表征出发，后续使用方式应按假设强度递增测试。

### A. Decoder-readable representation

最弱、已基本验证的先验：

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

下一步优先验证的先验：

```text
z 不只是能生成文本，还能区分当前 prefix 下正确和错误的 next step。
```

可用方式：

```text
candidate ranking
latent verifier
retrieval / reranking
search-time critic
```

这一步不要求 `z_i -> z_{i+1}` 可学，也不要求 continuous rollout 成立。

### C. Trajectory-structured representation

更强的先验：

```text
同一道题的 z_1, z_2, ... 具有局部轨迹结构。
```

可用方式：

```text
adjacent latent analysis
delta latent analysis
nearest-neighbor transition
local residual update
```

### D. Learnable single-step transition

再强的先验：

```text
存在函数 T，使 T(z_i) 接近 z_{i+1}，并且输出仍可被 decoder 读取。
```

可用方式：

```text
single-step transition
delta / residual transition
teacher-forced latent prediction
```

### E. Multi-step continuous rollout

最强、目前证据最不足的先验：

```text
重复应用 T 后，latent 不快速漂移，并能最终生成正确 answer。
```

这不应作为当前下一步主目标。

## 5. 当前实验轴

### Exp1: codec initialization

测试：

```text
NTP-init + standard codec objective
MTP-init + standard codec objective
```

目的：

```text
只改变初始化，判断 MTP init 是否让 sentence-level predictive codec 更好学。
```

### Exp2A: decoder token-level MTP

测试：

```text
decoder hidden state 同时预测多个 future tokens
```

目的：

```text
判断 token-level multi-horizon supervision 是否改善 decoder 局部语言建模或间接改善 z。
```

注意：这不是 step-level MTP。它仍然只训练当前 target。

### SFT sanity baselines

测试：

```text
next_step: question + previous_steps -> current_step
answer_from_steps: question + all_steps -> answer
answer_from_question: question -> answer
```

目的：

```text
判断任务本身对当前 GPT-2 是否过难；如果普通 SFT 都学不会，latent codec 更难学是预期内结果。
```

### Transition experiments

当前 transition 代码和配置存在，但研究上应降级为诊断对象，而不是主线。

使用 transition 结果时，必须先回答：

```text
失败是 transition 模型失败，还是 codec latent 本身不适合作为 transition state？
```

## 6. 下一步研究计划

当前下一步不应继续优先优化 transition model，而应先做 representation usefulness map。

### 6.1 Next-step candidate ranking

Falsifiable claim:

```text
给定 prefix latent z，MTP-init codec 的 z 比 NTP-init codec 的 z 更能从候选 step 中选出 gold next step。
```

候选负例：

```text
cross-question random step
same-question non-current step
model-generated wrong step
hard negative with similar numbers or format
```

指标：

```text
top-1 accuracy
MRR
gold score
max negative score
margin
hard-negative error rate
```

### 6.2 Latent-as-verifier

Falsifiable claim:

```text
z 可以作为当前 prefix 的 next-step verifier signal。
```

任务：

```text
(prefix, candidate_step) -> valid_next_step / invalid_next_step
```

指标：

```text
AUC
accuracy
false positive examples
false negative examples
```

### 6.3 Gold latent geometry

Falsifiable claim:

```text
同一道题相邻 prefix latent 比非相邻 latent 或跨题 random latent 更接近。
```

指标：

```text
same-question adjacent cosine / mse
same-question non-adjacent cosine / mse
cross-question random cosine / mse
```

### 6.4 Decoder sensitivity

Falsifiable claim:

```text
decoder 对 latent 偏差有可测的容忍区间；如果 transition 输出落在该区间外，rollout 失败是预期结果。
```

指标：

```text
noise scale -> token_acc / exact_match
interpolation alpha -> token_acc / exact_match
latent cosine vs decode success
```

## 7. 决策规则

```text
如果 ranking/verifier 强，但 trajectory 弱：
    使用 z 做 retrieval、reranking、verifier 或 search critic。
    暂缓 continuous transition。

如果 ranking/verifier 强，trajectory 也强：
    再测试 single-step transition，优先 delta / residual / nearest-neighbor。

如果 ranking/verifier 弱，但 generation 强：
    z 主要是 decoder-readable generation state，不是 reasoning-discriminative state。
    应重新设计 objective，例如 step-level MTP 或 contrastive next-step objective。

如果 transition teacher-forced 好但 rollout 差：
    优先诊断 latent drift、decoder sensitivity 和 next_type stop failure。

如果 direct rollout 低而 decoder sensitivity 显示 manifold 很窄：
    不应继续盲目调 transition loss，应先改变接口或使用 reencode / verifier-style usage。
```

## 8. 当前最小研究链条

```text
MTP helps sentence-level codec
-> test whether z is discriminative for next-step choice
-> test whether z can verify candidate reasoning states
-> test whether z has trajectory geometry
-> only then test learnable single-step transition
-> only after that test multi-step continuous rollout
```

这条链条把已验证结果和未验证假设分开，避免再次把 transition rollout 当作默认可行的下一步。
