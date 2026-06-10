# Representation Usefulness 分析计划

当前已实现并完成初步诊断：

```text
next-step candidate ranking
latent-as-verifier
```

旧的 transition 表征分析代码，包括 gold latent geometry 和 decoder sensitivity，已经从当前 analysis 入口移除。原因是这些分析容易把当前研究重心重新带回 `z_i -> z_{i+1}` 和 transition-first 路线。

## 1. 当前问题

已经验证的能力是：

```text
Encoder(prefix) -> z
Decoder(z) -> current step / answer
```

已经得到的回答：

```text
z 可以作为 reasoning-time candidate scorer。
z 可以作为初步 verifier signal。
MTP/NTP 在当前 GSM Exp1 诊断中基本持平。
```

而不是直接问：

```text
z 能不能被 transition model 递推？
```

## 2. 已完成诊断的 Claim

```text
给定 prefix latent z，Decoder(z) 的条件似然可以区分 gold next step 与负例候选。
```

当前结果支持这个更一般的 claim，但不支持 “MTP 明显优于 NTP”。

```text
如果某个 codec 的 sentence-level latent 更有 reasoning-discriminative value，
那么 Decoder(z) 对 gold next step 的条件似然应高于同题或跨题负例。
```

## 3. 实现口径

对每个 target position：

```text
prefix_i = question + s_1 + ... + s_{i-1}
gold_i = s_i 或 answer
z_i = Encoder(prefix_i)
```

构造候选集合：

```text
C_i = {gold_i} + same-question negatives + cross-question negatives + hard negatives
```

当前 negative 类型：

```text
same_question:
    同一道题中不是当前 target 的其他 step / answer。

cross_question:
    其他题中的 step / answer。

hard_wrong_*:
    由 gold target 直接扰动得到的局部相似错误候选。
```

当前 hard negatives 已实现，但仍偏“算式内部不一致”。如果继续做 verifier 诊断，下一轮应改成内部自洽但上下文错误的 hard negatives。

## 4. Score 定义

候选分数定义为：

```text
score(z_i, c) = - mean_token_NLL(c | z_i)
```

计算步骤：

```text
1. 用 Encoder(prefix_i) 得到 z_i。
2. 把 candidate 文本 c tokenized 为 y_1 ... y_T EOS。
3. 用 Decoder(z_i) teacher-forcing 计算每个 token 的 cross entropy。
4. 对有效 token 求平均 NLL。
5. 取负数作为 score。
```

解释：

```text
score 越高，说明 decoder 在 z_i 条件下越偏好该 candidate。
```

这个实验测的是：

```text
z_i 通过 decoder readout 是否能把 gold next step 排在负例前面。
```

它不测：

```text
candidate step embedding 是否与 z_i 对齐。
z_i 是否能递推到 z_{i+1}。
transition model 是否可学。
continuous rollout 是否可行。
```

## 5. 比较方式

同一 split、同一 `max_samples`、同一 negative 构造规则下，分别跑：

```text
NTP-init + standard objective
MTP-init + standard objective
```

当前只做 Exp1，因为 Exp2A 暂时没有可用的 codec checkpoint。

当前 GSM 配置：

```text
configs/analysis_ranking_exp1_gsm_ntp.yaml
configs/analysis_ranking_exp1_gsm_mtp.yaml
```

当前脚本：

```bash
bash scripts/analyze_ranking_exp1_gsm_ntp.sh
bash scripts/analyze_ranking_exp1_gsm_mtp.sh
bash scripts/analyze_ranking_gsm_all.sh
```

## 6. 指标

```text
top1_accuracy
mrr
gold_score
max_negative_score
gold_minus_max_negative_margin
negative_type_summary
examples
```

解释：

```text
top1_accuracy 高：
    gold 通常被排第一。

MRR 高：
    gold 即使没排第一，也通常靠前。

gold_minus_max_negative_margin 大：
    gold 比最强负例有稳定分离。

same_question margin 大：
    更支持 reasoning-discriminative signal。

cross_question margin 大但 same_question margin 小：
    可能主要是题目或表面格式匹配。
```

## 7. 输出

输出文件：

```text
outputs/analysis/candidate_ranking/gsm/<experiment_name>_valid_candidate_ranking.json
```

JSON 包含：

```text
implementation_contract
sample_count
target_pool_size
top1_accuracy
mrr
candidate_count
gold_score
max_negative_score
gold_minus_max_negative_margin
negative_type_summary
examples
```

`examples` 中会优先保存失败样例，也会保存少量早期样例，便于人工检查：

```text
prefix_text
target_kind
gold_text
rank
best_negative_type
top_candidates
```

## 8. Latent-as-verifier

6.2 的目标不是训练新的 verifier probe，而是先测试现有 codec readout 是否能直接提供 verifier 分数。

二分类样本：

```text
positive pair:
    (prefix_i, gold_i)

negative pair:
    (prefix_i, perturbed gold_i hard negative)
    (prefix_i, same-question non-gold candidate)
    (prefix_i, cross-question non-gold candidate)
```

hard negative 由 gold target 直接构造：

```text
hard_wrong_result:
    <<expr=result>> -> <<expr=result+delta>>

hard_wrong_operator:
    <<a+b=result>> -> <<a-b=result>>

hard_wrong_operand:
    <<a+b=result>> -> <<a+1+b=result>>

hard_wrong_answer:
    answer -> answer+delta
```

这些负例的作用是测试：

```text
Decoder(z) 的条件似然能否拒绝“格式正确、局部相似、但推理错误”的候选。
```

分数仍然使用：

```text
score(z_i, c) = - mean_token_NLL(c | z_i)
```

判定规则：

```text
candidate is valid if score >= threshold
```

当前第一版用 post-hoc best threshold：

```text
在当前 analysis split 上选择 accuracy 最高的 threshold。
```

这不是正式泛化评估，只用来判断 score 本身是否具有 verifier 可分性。后续如果信号强，再做 train/valid threshold split 或训练轻量 probe。

指标：

```text
AUC
best-threshold accuracy / precision / recall / F1
positive score summary
negative score summary
false_positive_rate by negative type
false_negative_rate by candidate type
failure examples
```

当前配置：

```text
configs/analysis_verifier_exp1_gsm_ntp.yaml
configs/analysis_verifier_exp1_gsm_mtp.yaml
```

当前脚本：

```bash
bash scripts/analyze_verifier_exp1_gsm_ntp.sh
bash scripts/analyze_verifier_exp1_gsm_mtp.sh
bash scripts/analyze_verifier_gsm_all.sh
```

解释边界：

```text
如果 AUC 高，说明 Decoder(z) 的条件似然可作为 verifier signal。
如果 same-question FPR 高，说明 verifier 仍容易接受同题错误步骤。
如果 cross-question FPR 低但 same-question FPR 高，说明它主要能排除不相关候选，还不能可靠判断 reasoning correctness。
如果 hard_wrong_* FPR 高，说明它还不能可靠拒绝局部相似的算术/变量绑定错误。
```

## 9. 下一轮：使用 z

下一步不再继续只做诊断，而是把 `z` 用进 inference/search 流程：

```text
reranking
verifier-gated generation
search-time critic
```

### 9.1 Reranking

```text
1. 对每个 prefix 产生 K 个 candidate steps。
2. 用 score(z, candidate) 打分。
3. 选择最高分 candidate。
4. 比较 reranked output 与 base generation。
```

### 9.2 Verifier-gated generation

```text
1. 生成 candidate step。
2. 用 score(z, candidate) 判断是否接受。
3. score 低则重新采样或回退。
4. 在 valid/test 上比较错误率。
```

### 9.3 Search-time critic

```text
1. beam/tree search 生成多个 partial traces。
2. 每一步用 score(Encoder(prefix), candidate_step) 做局部 critic。
3. 保留高分路径。
4. 比较 final answer accuracy。
```
