# Multi-Token Prediction for Latent Reasoning

本仓库实现了一个围绕 `doc/mtp_latent_reasoning_experiment_design.md` 的实验框架，目标是把 MTP 在 sentence-level latent reasoning 中的作用拆成可单独验证的实验轴，而不是把 MTP 与 latent reasoning 后训练混在一起比较。

当前代码优先覆盖文档中的第一优先级主线：

- `A1`: `NTP-init + standard codec objective`
- `A2`: `MTP-init + standard codec objective`
- `C1`: `NTP-init + step-level MTP codec objective`
- `C2`: `MTP-init + step-level MTP codec objective`

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

## 快速校准数据接口

先用小样本检查 `dataset/` 中的真实数据是否被正确展开：

```bash
bash scripts/inspect_prosqa_quick.sh
bash scripts/inspect_gsm_quick.sh
```

## 运行

当前 codec 已改为 GPT-2 架构：

- `encode`: GPT-2 encoder 取最后一个有效 token hidden state，再投影为 latent
- `decode`: 把 latent 投影成一个 prefix embedding，拼到 GPT-2 decoder 输入前面

先跑小样本：

```bash
bash scripts/train_codec_prosqa_quick.sh
bash scripts/train_transition_prosqa_quick.sh

bash scripts/train_codec_gsm_quick.sh
bash scripts/train_transition_gsm_quick.sh
```

训练监控默认写到各自 `output_dir/tensorboard/`。查看方式：

```bash
tensorboard --logdir outputs
```

再跑完整配置：

```bash
PYTHONPATH=src python -m mtp_latent.cli train-codec --config configs/prosqa_a1.yaml
```

```bash
PYTHONPATH=src python -m mtp_latent.cli train-transition \
  --config configs/prosqa_a1.yaml \
  --codec-checkpoint outputs/prosqa_a1/codec_best.pt
```

```bash
PYTHONPATH=src python -m mtp_latent.cli evaluate \
  --config configs/prosqa_a1.yaml \
  --codec-checkpoint outputs/prosqa_a1/codec_best.pt \
  --transition-checkpoint outputs/prosqa_a1/transition_best.pt
```

## 初始化与扩展

- `model.init_checkpoint` 对应 codec 初始化来源，可填 NTP 或 MTP 预训练权重。
- `transition.init_checkpoint` 对应 transition 初始化来源，可填 random / NTP / MTP。
- `codec_objective.name` 支持：
  - `standard`: 只训练 horizon-1。
  - `step_mtp`: 同时训练 horizon-1/2/3。

当前实现是 GPT-2 codec + MLP transition，适合先把实验逻辑、数据接口、指标与对照关系跑通。后续如果你要替换成你自己的 NTP/MTP checkpoint，只需要保持以下接口稳定：

- `ReasoningCodec.encode(prefix_ids, prefix_mask) -> z`
- `ReasoningCodec.decode(z, target_tokens) -> logits`
- `LatentTransitionModel(z_t) -> z_t+1`

更详细的代码说明见 [doc/code_experiment_guide.md](/Users/zhangjunyi/project/multi-token-predict-for-latent/doc/code_experiment_guide.md)。
