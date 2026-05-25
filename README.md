# Multi-Token Prediction for Latent Reasoning

本仓库实现了一个围绕 `doc/mtp_latent_reasoning_experiment_design.md` 的实验框架，目标是把 MTP 在 sentence-level latent reasoning 中的作用拆成可单独验证的实验轴，而不是把 MTP 与 latent reasoning 后训练混在一起比较。

当前仓库先只对齐设计文档中的实验一：

- `ProsQA + NTP-init`
- `ProsQA + MTP-init`
- `GSM + NTP-init`
- `GSM + MTP-init`

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
- `target_2 = next_step`
- `target_3 = next_next_step`

超出长度的 horizon 会自动 mask。

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

当前训练的是实验一里的 sentence-level predictive encoder-decoder：

- `encode`: GPT-2 encoder 取最后一个有效 token hidden state，再投影为 latent
- `decode`: 把 latent 投影成一个 prefix embedding，拼到 GPT-2 decoder 输入前面
- `train.device`: 支持 `auto`、`cuda`、`cuda:0`、`cpu`
- `torchrun + DDP`: 支持多卡并行；单卡命令语义保持不变

在服务器上先把下面四个 YAML 里的 `model.model_name_or_path` 改成实际模型路径：

```bash
configs/exp1_prosqa_ntp_init.yaml
configs/exp1_prosqa_mtp_init.yaml
configs/exp1_gsm_ntp_init.yaml
configs/exp1_gsm_mtp_init.yaml
```

然后直接运行对应脚本。默认会用 `torchrun` 按本机可见 GPU 数启动 DDP 多卡训练；如果只想用部分卡，可以手动指定 `NPROC_PER_NODE`：

```bash
NPROC_PER_NODE=4 bash scripts/train_exp1_prosqa_ntp_init.sh

bash scripts/train_exp1_prosqa_ntp_init.sh
bash scripts/train_exp1_prosqa_mtp_init.sh
bash scripts/train_exp1_gsm_ntp_init.sh
bash scripts/train_exp1_gsm_mtp_init.sh
```

训练监控默认写到各自 `output_dir/tensorboard/`。查看方式：

```bash
bash scripts/tensorboard_exp1.sh
```

## 初始化与扩展

- 当前阶段只比较初始化来源，不训练 transition，也不做 step-level MTP objective。
- 实验一配置里的 `max_horizon=1`，只预测当前 step。
- 当前最重要的结果是四组实验的 `train loss` 和 `valid loss` TensorBoard 曲线。

当前实现是 GPT-2 encoder-decoder 训练框架，后续如果你要继续做实验二、实验三，再在这个基础上继续加 objective 和 transition 相关入口。

- `ReasoningCodec.encode(prefix_ids, prefix_mask) -> z`
- `ReasoningCodec.decode(z, target_tokens) -> logits`

更详细的代码说明见 [doc/code_experiment_guide.md](/Users/zhangjunyi/project/multi-token-predict-for-latent/doc/code_experiment_guide.md)。
