# Multi-Token Prediction for Latent Reasoning

本仓库研究 MTP 是否能帮助模型学出更有用的 sentence-level latent representation，并进一步探索这些 representation 如何服务 latent state reasoning。

当前主线不是直接假设 transition rollout 可行，而是：

```text
MTP helps sentence-level codec
-> test whether z is useful for ranking / verification / geometry
-> only then test transition and continuous rollout
```

## 当前研究状态

已经验证的方向：

```text
Encoder(prefix) -> z
Decoder(z) -> current step / answer
```

当前本地 `output/codec_history.json` 显示，GSM codec 训练已经能达到较高的 token / answer 指标，`output/epoch_025.json` 中的 generation samples 也能正常以 EOS 结束。

当前 claim boundary：

```text
可以说：z 是 decoder-readable 的 sentence-level latent。
不能说：z 已经是可递推 transition state。
```

详细研究状态见：

```text
doc/mtp_latent_reasoning_experiment_design.md
doc/representation_usefulness_analysis_plan.md
```

## 数据格式

每条样本是 reasoning trace：

```json
{"question": "...", "steps": ["step 1", "step 2"], "answer": "..."}
```

代码会展开为：

```text
prefix = question + previous_steps
target = current_step 或 answer
target_kind = step 或 answer
```

当前 codec 已经把 `answer` 作为 trace 最后一个 target。

## 目录结构

```text
configs/                实验配置
dataset/                数据
doc/                    当前文档
doc/archive/            历史记录和过期计划
scripts/                训练与分析脚本
src/mtp_latent/         核心代码
output/                 本地临时输出样例
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 可运行实验

### Exp1: standard codec

```bash
bash scripts/train_exp1_prosqa_ntp_init.sh
bash scripts/train_exp1_prosqa_mtp_init.sh
bash scripts/train_exp1_gsm_ntp_init.sh
bash scripts/train_exp1_gsm_mtp_init.sh
```

### Exp2A: decoder token-level MTP

```bash
bash scripts/train_exp2a_prosqa_ntp_init.sh
bash scripts/train_exp2a_prosqa_mtp_init.sh
bash scripts/train_exp2a_gsm_ntp_init.sh
bash scripts/train_exp2a_gsm_mtp_init.sh
```

### Candidate ranking analysis

```bash
bash scripts/analyze_ranking_gsm_all.sh
```

### Latent verifier analysis

```bash
bash scripts/analyze_verifier_gsm_all.sh
```

### SFT sanity baselines

```bash
bash scripts/train_sft_prosqa_next_step_ntp.sh
bash scripts/train_sft_prosqa_next_step_mtp.sh
bash scripts/train_sft_prosqa_answer_from_steps_ntp.sh
bash scripts/train_sft_prosqa_answer_from_steps_mtp.sh
bash scripts/train_sft_prosqa_answer_from_question_ntp.sh
bash scripts/train_sft_prosqa_answer_from_question_mtp.sh
```

## 输出指标

Codec valid 主要看：

```text
valid loss
token_h1_loss
token_h1_acc
answer_acc
answer_token_loss
answer_token_acc
```

Generation samples 包含：

```text
prefix_text
target_kind
target_text
predicted_text
finished_with_eos
answer_correct
```

Transition valid 主要看：

```text
teacher_forced_answer_acc
rollout_direct_answer_acc
rollout_direct_answer_stop_rate
rollout_reencode_answer_acc
rollout_reencode_answer_stop_rate
pred_vs_target_latent_cosine
pred_vs_target_latent_mse
```

## 当前下一步

优先做 representation usefulness map：

```text
1. next-step candidate ranking
2. latent-as-verifier
3. hard negative ranking
4. candidate embedding alignment
5. transition diagnostics
```

这一步的目标是回答：

```text
学到的 sentence-level latent z 应该如何被使用？
```

而不是继续默认：

```text
z_i -> z_{i+1} transition 一定是正确接口。
```

更详细的代码说明见：

```text
doc/code_experiment_guide.md
```
