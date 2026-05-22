# 代码实验说明

本文档解释当前代码如何对应 `mtp_latent_reasoning_experiment_design.md` 的实验设计，并说明现在的实现已经切换为 GPT-2 架构、真实 `dataset/` 数据接口以及 TensorBoard 监控。

## 1. 当前实现状态

当前仓库的实验主线已经从最初的轻量占位实现，更新为：

- GPT-2 风格 codec
- `prosqa` 与 `gsm` 的真实数据文件接入
- 小样本 quick 配置用于数据接口校准
- TensorBoard 训练监控

核心目标没有变化，仍然是把文档中的实验轴拆开：

- `codec initialization`
- `codec objective`
- `transition learning`
- `rollout mode`

## 2. 数据接口

### 2.1 当前读取的数据

当前直接读取仓库中的：

- `dataset/prosqa_train.json`
- `dataset/prosqa_valid.json`
- `dataset/prosqa_test.json`
- `dataset/gsm_train.json`
- `dataset/gsm_valid.json`
- `dataset/gsm_test.json`

代码同时兼容：

- JSON 数组文件
- JSONL 文件

### 2.2 样本字段

当前数据至少要求：

```json
{"question": "...", "steps": ["..."], "answer": "..."}
```

其中：

- `question` 是前缀问题
- `steps` 是 reasoning trace
- `answer` 是最终答案

对于 `gsm` 中可能出现的空 step，loader 会按配置自动过滤。

### 2.3 trace 展开方式

`src/mtp_latent/data.py` 实现了文档第 4 节的统一展开逻辑：

- `prefix = question + previous_steps`
- `target_1 = s_i`
- `target_2 = s_i+1`
- `target_3 = s_i+2`

超出 trace 长度的 horizon 使用 `horizon_mask` 屏蔽。

### 2.4 小样本校准

为了先对齐真实数据接口，配置中增加了：

- `train_max_records`
- `valid_max_records`
- `test_max_records`

你可以先跑：

```bash
bash scripts/inspect_prosqa_quick.sh
bash scripts/inspect_gsm_quick.sh
```

它们会调用 `inspect-data`，打印：

- 当前读取到的 record 数
- 展开后的 sample 数
- 第一条 prefix
- 第一组 future steps
- dataloader batch 数

## 3. 模型结构

### 3.1 GPT-2 codec

`src/mtp_latent/models.py` 中的 `ReasoningCodec` 已经改成 GPT-2 架构。

当前做法是：

1. `encode`：
   - 用 GPT-2 encoder 编码 prefix
   - 取最后一个有效 token 的 hidden state
   - 线性投影成 latent `z`

2. `decode`：
   - 把 latent `z` 投影成一个 decoder prefix embedding
   - 拼接到 target token embedding 前
   - 用 GPT-2 LM head 预测 step token

这对应 sentence-level codec 的最小可运行版本。

### 3.2 transition model

`LatentTransitionModel` 目前仍是 MLP：

- 输入 `z_t`
- 输出 `z_t+1`

它的作用是先把实验三的对照跑通，不把 transition 结构复杂化。

### 3.3 为什么现在这样实现

当前实现优先考虑三件事：

1. 和你后续接真实 NTP/MTP checkpoint 的路径一致
2. 保留文档要求的实验变量可控性
3. 尽快把数据、训练、评估、监控全链路打通

因此目前不是“最终模型”，而是“可直接替换 backbone 的实验框架”。

## 4. 配置如何对应实验设计

### 4.1 初始化对照

以下配置用于实验一：

- `configs/prosqa_a1.yaml`
- `configs/prosqa_a2.yaml`

对应：

- `A1 = NTP-init + standard objective`
- `A2 = MTP-init + standard objective`

### 4.2 step-level MTP objective

以下配置用于实验二的关键主线：

- `configs/prosqa_c1.yaml`
- `configs/prosqa_c2.yaml`

对应：

- `C1 = NTP-init + step_mtp objective`
- `C2 = MTP-init + step_mtp objective`

### 4.3 quick 配置

为了先校准数据和训练入口，当前增加了：

- `configs/prosqa_a1_quick.yaml`
- `configs/gsm_a1_quick.yaml`

这两份只读取一小部分真实数据。

## 5. 训练与监控

### 5.1 CLI 命令

当前 `src/mtp_latent/cli.py` 支持：

- `inspect-data`
- `train-codec`
- `train-transition`
- `evaluate`
- `show-history`

### 5.2 TensorBoard

训练监控已接入 TensorBoard。

默认写入：

- `outputs/<experiment>/tensorboard/codec`
- `outputs/<experiment>/tensorboard/transition`

当前记录的主要内容包括：

- codec train step loss
- codec train epoch loss
- codec valid loss
- horizon token accuracy
- transition train loss
- transition valid loss
- retrieval accuracy
- positive / negative score
- margin

查看方式：

```bash
tensorboard --logdir outputs
```

### 5.3 输出文件

每次实验会在 `train.output_dir` 下产出：

- `codec_best.pt`
- `codec_history.json`
- `transition_best.pt`
- `transition_history.json`
- `evaluation.json`
- `tensorboard/`

## 6. 训练脚本

当前已经补好 shell 入口：

- `scripts/inspect_prosqa_quick.sh`
- `scripts/inspect_gsm_quick.sh`
- `scripts/train_codec_prosqa_quick.sh`
- `scripts/train_transition_prosqa_quick.sh`
- `scripts/train_codec_gsm_quick.sh`
- `scripts/train_transition_gsm_quick.sh`

这些脚本优先服务于：

- 数据接口校准
- 小样本 smoke test
- 远端快速验证命令可用性

## 7. 还没做满的部分

当前主线已经能支撑实验框架，但以下部分仍然是后续工作：

- decoder token-level MTP 专门 ablation
- 更完整的 multi-step rollout 评估
- answer-level 指标
- task-specific validity checker
- cross-task transfer 自动脚本
- 真实 NTP/MTP checkpoint 的定制权重映射逻辑

## 8. 你后续最可能继续改的地方

如果你下一步要接入自己的预训练权重，建议按这个顺序：

1. 在配置里把 `tokenizer_name_or_path` / `model_name_or_path` 指向你的本地 GPT-2 权重
2. 在 `load_init_checkpoint()` 中补 NTP/MTP checkpoint 的实际映射
3. 保持 `ReasoningCodec.encode/decode` 接口不变
4. 在 `evaluation.py` 中补多步 rollout 与 answer-level 评估
5. 再追加 Blocksworld / GSM8K 的任务特定分析
