# 代码实验说明：当前版

本文档只描述当前代码真实支持的实验接口和指标。历史规划和过期叙述已移到 `doc/archive/` 或主设计文档中。

## 1. 当前代码支持什么

当前 CLI 支持：

```text
train-codec
train-transition
train-sft
evaluate
inspect-data
show-history
analyze-transition-repr
```

当前主要可运行实验：

```text
Exp1: standard codec objective
Exp2A: decoder_token_mtp codec objective
Exp3: frozen codec + transition model
SFT: ordinary supervised baselines
Analysis: transition / representation diagnostics
```

## 2. 数据形式

每条样本是 reasoning trace：

```text
question
s_1
s_2
...
s_n
answer
```

`src/mtp_latent/data.py` 会把它展开为：

```text
prefix = question + previous_steps
target = current_step 或 answer
target_kind = step 或 answer
```

当前 codec 已经把 `answer` 纳入 target，并在训练和验证中统计 answer 相关指标。

## 3. Codec 模型接口

当前 codec 是 GPT-2 style encoder-decoder：

```text
Encoder(prefix) -> z
Decoder(z) -> target_text
```

训练时，target token 序列是：

```text
y_1, y_2, ..., y_T, EOS
```

decoder 输入是：

```text
latent_prefix + y_1, y_2, ..., y_T
```

监督目标是：

```text
y_1, y_2, ..., y_T, EOS
```

没有人工 BOS。第一个 token 由 latent 直接预测。padding label 使用 `-100`，所以真实 EOS 不会因为 `pad_token_id == eos_token_id` 被误忽略。

这保证了：

```text
latent 能直接启动 step / answer generation
decoder 学会何时停止
后续离散 rollout 有明确 EOS stop rule
```

## 4. Exp1: standard codec objective

目标：

```text
只改变初始化来源，比较 NTP-init 与 MTP-init 是否让 codec 更好学。
```

配置：

```text
configs/exp1_prosqa_ntp_init.yaml
configs/exp1_prosqa_mtp_init.yaml
configs/exp1_gsm_ntp_init.yaml
configs/exp1_gsm_mtp_init.yaml
```

共同设置：

```text
codec_objective.name = standard
data.max_horizon = 1
```

主要指标：

```text
train loss
valid loss
token_h1_loss
token_h1_acc
answer_acc
answer_token_loss
answer_token_acc
```

## 5. Exp2A: decoder token-level MTP

目标：

```text
测试 decoder hidden state 的 token-level multi-horizon supervision 是否有收益。
```

这不是 step-level MTP。它仍然只预测当前 target，只是在 target token 序列内部增加未来 token 监督。

设 target 为：

```text
y_1, y_2, ..., y_T, EOS
```

则默认监督为：

```text
h_t -> y_{t+1}
h_t -> y_{t+2}
h_t -> y_{t+3}
```

配置：

```text
configs/exp2a_prosqa_ntp_init.yaml
configs/exp2a_prosqa_mtp_init.yaml
configs/exp2a_gsm_ntp_init.yaml
configs/exp2a_gsm_mtp_init.yaml
```

关键字段：

```yaml
codec_objective:
  name: decoder_token_mtp
  token_prediction_horizons: [1, 2, 3]
  token_prediction_weights: [1.0, 0.5, 0.25]
```

主要指标：

```text
token_h1_loss / token_h1_acc
token_h2_loss / token_h2_acc
token_h3_loss / token_h3_acc
answer_acc
```

## 6. Answer supervision

当前 data loader 会把 answer 作为 trace 最后一个 target：

```text
question + s_1 + ... + s_n -> answer
```

训练支持：

```yaml
codec_objective:
  answer_loss_weight: 2.0
```

当 `answer_loss_weight > 1.0` 时，answer 样本的 loss 会被放大，以缓解每条 trace 只有一个 answer target 的样本占比问题。

valid 阶段会额外统计：

```text
answer_acc
answer_count
answer_token_loss
answer_token_acc
```

其中：

```text
answer_acc = free generation exact match
answer_token_acc = teacher-forced answer token accuracy
```

## 7. Valid generation 输出

codec 每个 epoch 会保存生成样例：

```text
outputs/<experiment_name>/valid_generations/epoch_001.json
```

每个样例包含：

```text
prefix_text
target_kind
target_text
predicted_text
finished_with_eos
answer_correct
```

本地临时 `output/epoch_025.json` 是同类 compact 样例文件。

## 8. Compact history

训练会保存紧凑指标摘要：

```text
codec_valid_compact.json
transition_valid_compact.json
sft_valid_compact.json
```

通常包含：

```text
latest
best_by_loss
best_by_metric
recent_epochs
```

当前本地 `output/codec_history.json` 是完整 history 文件，不是 compact 摘要。

## 9. Transition 当前接口

transition 代码支持 frozen codec 后的 mixed-sequence training：

```text
[q_1, ..., q_n, z_1, z_2, ..., z_m]
```

监督位置：

```text
q_n -> s_1
z_1 -> s_2
...
z_m -> answer
```

transition valid 记录：

```text
teacher_forced_answer_acc
rollout_direct_answer_acc
rollout_direct_answer_stop_rate
rollout_reencode_answer_acc
rollout_reencode_answer_stop_rate
```

还可选 latent auxiliary loss：

```text
cosine
cosine_huber
infonce
infonce_huber
```

对应指标：

```text
latent_loss
infonce_loss
latent_huber_loss
pred_vs_target_latent_cosine
pred_vs_target_latent_mse
```

研究解释时要注意：transition 实现存在不等于 transition 假设成立。当前主线应先验证 representation usefulness，再决定是否继续强化 transition。

## 10. SFT sanity baselines

ProsQA 当前有三类 SFT baseline：

```text
next_step: question + previous_steps -> current_step
answer_from_steps: question + all_steps -> answer
answer_from_question: question -> answer
```

配置：

```text
configs/sft_prosqa_next_step_ntp.yaml
configs/sft_prosqa_next_step_mtp.yaml
configs/sft_prosqa_answer_from_steps_ntp.yaml
configs/sft_prosqa_answer_from_steps_mtp.yaml
configs/sft_prosqa_answer_from_question_ntp.yaml
configs/sft_prosqa_answer_from_question_mtp.yaml
```

指标：

```text
token_loss
token_acc
step_acc
answer_acc
```

## 11. 常用命令

训练 codec：

```bash
bash scripts/train_exp1_gsm_ntp_init.sh
bash scripts/train_exp1_gsm_mtp_init.sh
bash scripts/train_exp2a_gsm_ntp_init.sh
bash scripts/train_exp2a_gsm_mtp_init.sh
```

训练 transition：

```bash
bash scripts/train_exp3_transition_exp1_gsm_ntp.sh
```

表征分析：

```bash
bash scripts/analyze_transition_exp1_gsm_ntp_repr.sh
```

TensorBoard：

```bash
bash scripts/tensorboard_exp1.sh
bash scripts/tensorboard_exp2a.sh
bash scripts/tensorboard_exp3.sh
```

## 12. 当前解释边界

可由当前 codec 结果支持的结论：

```text
latent z 是 decoder-readable 的。
z 能支持 step / answer generation。
EOS 接口已经可用。
answer supervision 已经实现。
```

仍需诊断的结论：

```text
z 是否能区分正确 next step 与 hard negatives。
z 是否能作为 verifier / ranker。
z 是否有局部 trajectory geometry。
z 是否适合单步 transition。
z 是否适合多步 continuous rollout。
```
