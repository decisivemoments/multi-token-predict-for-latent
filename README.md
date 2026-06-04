# Multi-Token Prediction for Latent Reasoning

本仓库实现了一个围绕 `doc/mtp_latent_reasoning_experiment_design.md` 的实验框架，目标是把 MTP 在 sentence-level latent reasoning 中的作用拆成可单独验证的实验轴，而不是把 MTP 与 latent reasoning 后训练混在一起比较。

当前仓库当前对齐设计文档中的两组实验：

- `ProsQA + NTP-init`
- `ProsQA + MTP-init`
- `GSM + NTP-init`
- `GSM + MTP-init`

分别覆盖：

- 实验一：`standard` objective
- 实验二A：`decoder_token_mtp` objective
- 实验三当前版本：`transition model + next_type(step/answer) + frozen decoder supervision`

## 数据格式

当前代码同时支持两种格式：

- JSON 数组文件
- JSONL 文件

每条样本至少包含：

```json
{"question": "...", "steps": ["step 1", "step 2", "step 3"], "answer": "..."}
```

代码会自动把每条 trace 展开成：

- `prefix = question + previous_steps`
- `target_1 = current_step`

实验一和实验二A都固定 `data.max_horizon = 1`，因此这里只监督 `current_step`。实验二A的多未来预测发生在 `current_step` 的 token 序列内部，而不是 step 级别。

## 目录结构

```text
configs/                实验配置模板
doc/                    设计文档与代码说明
src/mtp_latent/         核心代码
mutagen.yml             本机/服务器同步配置模板
requirements.txt        Python 依赖
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

当前训练的是 sentence-level predictive encoder-decoder：

- `encode`: GPT-2 encoder 取最后一个有效 token hidden state，再投影为 latent
- `decode`: 把 latent 投影成一个 prefix embedding，拼到 GPT-2 decoder 输入前面
- `train.device`: 支持 `auto`、`cuda`、`cuda:0`、`cpu`
- `torchrun + DDP`: 支持多卡并行；单卡命令语义保持不变
- `train-transition`: 支持基于冻结 codec 的 transition 训练
- `train-sft`: 支持 ProsQA/GSM 风格的普通 SFT sanity check

在服务器上先把对应 YAML 里的 `model.tokenizer_name_or_path` 和 `model.model_name_or_path` 改成实际模型路径。transition 训练同时还要求 `transition.codec_checkpoint` 指向实验一或实验二A输出的 `codec_best.pt`。

实验一配置：

```bash
configs/exp1_prosqa_ntp_init.yaml
configs/exp1_prosqa_mtp_init.yaml
configs/exp1_gsm_ntp_init.yaml
configs/exp1_gsm_mtp_init.yaml
```

实验二A配置：

```bash
configs/exp2a_prosqa_ntp_init.yaml
configs/exp2a_prosqa_mtp_init.yaml
configs/exp2a_gsm_ntp_init.yaml
configs/exp2a_gsm_mtp_init.yaml
```

然后直接运行对应脚本。默认会用 `torchrun` 按本机可见 GPU 数启动 DDP 多卡训练；如果只想用部分卡，可以手动指定 `NPROC_PER_NODE`：

```bash
NPROC_PER_NODE=4 bash scripts/train_exp1_prosqa_ntp_init.sh

bash scripts/train_exp1_prosqa_ntp_init.sh
bash scripts/train_exp1_prosqa_mtp_init.sh
bash scripts/train_exp1_gsm_ntp_init.sh
bash scripts/train_exp1_gsm_mtp_init.sh

bash scripts/train_exp2a_prosqa_ntp_init.sh
bash scripts/train_exp2a_prosqa_mtp_init.sh
bash scripts/train_exp2a_gsm_ntp_init.sh
bash scripts/train_exp2a_gsm_mtp_init.sh
```

训练监控默认写到各自 `output_dir/tensorboard/`。查看方式：

```bash
bash scripts/tensorboard_exp1.sh
bash scripts/tensorboard_exp2a.sh
bash scripts/tensorboard_exp3.sh
```

每个 epoch 的 valid 生成样例会额外写到：

```bash
outputs/<experiment_name>/valid_generations/epoch_001.json
```

里面会保存 `prefix_text`、`target_kind`、`target_text`、`predicted_text`、`finished_with_eos` 和 `answer_correct`。同时，`valid_metrics` 里会额外包含 `answer_acc`，方便直接看 answer 样本上的生成准确率。
现在还会额外包含：

- `answer_token_loss`
- `answer_token_acc`

用于区分“answer token 本身有没有学会”和“自由生成 exact match 为什么仍然偏低”。

另外还会保存一个紧凑版 valid 摘要：

```bash
outputs/<experiment_name>/codec_valid_compact.json
```

这个文件每个 epoch 只保留少量关键字段，方便直接复制文本做外部分析。

transition 训练完成后，也会额外写：

```bash
outputs/<experiment_name>/transition_valid_generations/epoch_001.json
```

里面会保存两类内容：

- `samples`：position-level teacher-forcing 检查，包含 `gold_type`、`predicted_type`、`gold_text`、`predicted_text` 和 `finished_with_eos`
- `rollout_samples`：question-only rollout 检查，包含
  - `teacher_forced_answer_prediction`
  - `rollout_direct`
  - `rollout_reencode`

transition valid 现在还会额外记录三类 answer 指标：

- `teacher_forced_answer_acc`
- `rollout_direct_answer_acc`
- `rollout_reencode_answer_acc`

以及对应的 `*_answer_stop_rate`，用来衡量 rollout 时模型有没有真的切换到 `answer`。

transition 也会保存紧凑版 valid 摘要：

```bash
outputs/<experiment_name>/transition_valid_compact.json
```

这个文件同样只保留关键指标，适合直接复制给 Codex 或其他模型分析。

transition 训练的初始化口径现在是固定的：

- frozen codec 总是从 `transition.codec_checkpoint` 加载
- transition backbone 总是从 `model.model_name_or_path` 加载预训练 GPT-2
- `transition.init_checkpoint` 只用于恢复一个已经训练中的 transition checkpoint

另外还补了三种 SFT baseline，用来判断任务本身对当前 GPT-2 是否已经过难：

- `next_step`: `question + previous_steps -> current_step`
- `answer_from_steps`: `question + all_steps -> answer`
- `answer_from_question`: `question -> answer`

当前先提供了 ProsQA 的 6 组现成配置和脚本：

```bash
configs/sft_prosqa_next_step_ntp.yaml
configs/sft_prosqa_next_step_mtp.yaml
configs/sft_prosqa_answer_from_steps_ntp.yaml
configs/sft_prosqa_answer_from_steps_mtp.yaml
configs/sft_prosqa_answer_from_question_ntp.yaml
configs/sft_prosqa_answer_from_question_mtp.yaml
```

## 初始化与扩展

- 当前阶段已经支持 codec 训练和 transition 训练，但还没有实现 step-level MTP objective。
- codec 训练当前会把 `answer` 也作为 trace 的最后一个 target。
- 实验一和实验二A都固定 `max_horizon=1`，只预测当前 target。
- 实验二A额外在 decoder hidden state 上预测 `current token / next token / next next token`。
- 当前最重要的结果是各组实验的 `train loss`、`valid loss` 和 token-horizon 指标 TensorBoard 曲线。

当前实现是 GPT-2 encoder-decoder 训练框架，后续如果你要继续做实验二、实验三，再在这个基础上继续加 objective 和 transition 相关入口。

- `ReasoningCodec.encode(prefix_ids, prefix_mask) -> z`
- `ReasoningCodec.decode(z, target_tokens) -> logits`

更详细的代码说明见 [doc/code_experiment_guide.md](/Users/zhangjunyi/project/multi-token-predict-for-latent/doc/code_experiment_guide.md)。
