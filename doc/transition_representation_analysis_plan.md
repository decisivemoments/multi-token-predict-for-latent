# Representation Usefulness 分析计划

当前不再把下一步直接定义为“优化 transition model”。更稳妥的问题是：

```text
已经学到的 sentence-level latent z，可以怎样被可靠使用？
```

本计划把原来的 transition representation analysis 扩展为 representation usefulness map。

## 1. 已知状态

当前 codec 已经能：

```text
Encoder(prefix) -> z
Decoder(z) -> current step / answer
```

本地 `output/codec_history.json` 和 `output/epoch_025.json` 显示：

```text
best valid token_h1_acc: 0.8784
best valid answer_acc: 0.8684
generation samples finish with EOS
```

因此下一步不是先假设 `z_i -> z_{i+1}` 可学，而是先测试 `z` 在 reasoning-time 是否有判别性和可用性。

## 2. 分析一：next-step candidate ranking

问题：

```text
给定 prefix latent z，能否从候选 step 中选出 gold next step？
```

候选集合：

```text
gold current step
cross-question random step
same-question non-current step
model-generated wrong step
hard negative with similar numbers / operators / format
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

要比较：

```text
NTP-init codec vs MTP-init codec
standard objective vs decoder_token_mtp objective
```

解释：

```text
如果 ranking 强，z 可以先作为 retriever / reranker / search critic 使用。
如果 ranking 只在 easy negatives 上强，z 可能主要学到 surface continuation。
如果 ranking 弱但 generation 好，z 更像 decoder-readable generation state，而不是 reasoning-discriminative state。
```

## 3. 分析二：latent-as-verifier

问题：

```text
z 能否判断 candidate step 是否是当前 prefix 的合理下一步？
```

任务：

```text
(prefix, candidate_step) -> valid_next_step / invalid_next_step
```

负例类型：

```text
random wrong step
same-question wrong step
wrong arithmetic step
right format but wrong variable step
model-generated wrong step
```

指标：

```text
AUC
accuracy
false positive rate by negative type
false negative examples
```

解释：

```text
如果 verifier 强，可以用 z 改善 latent state reasoning 的 search / reranking 部分。
这比 continuous transition 需要更少假设。
```

## 4. 分析三：gold latent geometry

问题：

```text
同一道题里的 z_1, z_2, z_3 是否构成局部轨迹？
```

统计：

```text
same-question adjacent latent cosine / mse
same-question non-adjacent latent cosine / mse
cross-question random latent cosine / mse
```

解释：

```text
如果 adjacent 不比 random 更近，绝对 latent transition 不是自然任务。
如果 adjacent 明显更近，可以继续测试 delta / residual / nearest-neighbor transition。
```

## 5. 分析四：decoder sensitivity

问题：

```text
decoder 对 latent 偏差的容忍区间有多大？
```

统计：

```text
gold latent baseline decode token_acc / exact_match
Gaussian noise scale -> decode degradation
random latent interpolation alpha -> decode degradation
latent cosine vs decode success
```

解释：

```text
如果 decoder 对小扰动极敏感，transition 输出必须非常贴近 manifold。
如果 decoder 容忍区间较宽，transition 失败更可能来自 state prediction 本身。
```

## 6. 分析五：transition error decomposition

只有在前四项有足够证据后，再分析 transition。

问题：

```text
transition 失败来自哪一层？
```

分解：

```text
teacher-forced decode success
next_type step/answer prediction
direct rollout answer stop rate
direct rollout answer acc
reencode rollout answer acc
rollout depth bucket
step index bucket
pred_vs_target_latent_cosine / mse
```

解释：

```text
teacher-forced 好但 direct rollout 差：递推稳定性或 exposure bias 问题。
direct stop rate 低：next_type / termination 问题。
latent cosine 高但 decode 差：decoder manifold sensitivity 问题。
latent cosine 低且 decode 差：transition state prediction 问题。
```

## 7. 推荐实现顺序

```text
1. gold latent geometry
2. decoder sensitivity
3. next-step candidate ranking
4. latent-as-verifier
5. transition error decomposition
```

当前已有 `configs/analysis_transition_exp1_gsm_ntp.yaml` 和脚本：

```bash
bash scripts/analyze_transition_exp1_gsm_ntp_repr.sh
```

它们先覆盖 geometry 和 sensitivity。后续应把 ranking / verifier 加入同一 analysis 目录，而不是继续新开 transition-first 实验。
