# 代码实验说明

本文档当前对齐两个阶段：

1. 实验一：MTP initialization 是否让 codec 更好学
2. 实验二A：decoder token-level MTP objective 是否有收益

这里仍然不展开实验二B、实验三、rollout 或 transfer。

## 1. 当前要验证什么

当前代码支持两类对照：

### 1.1 实验一：只改初始化

下面这个对照保持不变：

- 同一个模型结构
- 同一个训练目标
- 同一个数据集
- 同一个训练预算
- 唯一差别是初始化模型权重不同

对应设计文档中的两组条件：

- `A1 = NTP-init + standard objective`
- `A2 = MTP-init + standard objective`

也就是说，当前阶段的代码目标不是“做完整 latent reasoning pipeline”，而是先回答：

> 在 sentence-level predictive encoder-decoder 训练里，MTP 初始化是否比 NTP 初始化更容易训练？

### 1.2 实验二A：只改 decoder token-level objective

实验二A在实验一的基础上，再单独回答：

> 如果 decoder 从同一个 hidden state 同时预测多个未来 token，而不是只预测下一个 token，是否会改善训练与下游表征质量？

实验二A对应设计文档里的：

- `B1 = NTP-init + decoder token-level MTP codec objective`
- `B2 = MTP-init + decoder token-level MTP codec objective`

## 2. 当前实验边界

### 2.1 模型

当前训练的是一个 GPT-2 架构下的 encoder-decoder 式 sentence-level predictive model：

- `Encoder(x_i) -> z_i`
- `Decoder(z_i) -> s_i`

其中：

- `x_i = question + previous steps`
- `s_i = 当前 reasoning step`

这里的重点不是模型形式是否最终最优，而是保证实验一的变量控制成立。

### 2.2 唯一变量

实验一里唯一允许变化的是初始化来源：

- NTP GPT-2 权重
- MTP GPT-2 权重

你会把服务器上的模型路径直接写进 YAML。

当前实现约定如下：

- `model.model_name_or_path`：填写服务器上的 GPT-2 初始化目录
- `model.init_checkpoint`：默认留空

也就是说，实验一优先使用 `model_name_or_path` 作为初始化来源，而不是再额外叠加第二层 checkpoint。

### 2.3 数据集

当前实验一只使用两个数据集：

- `ProsQA`
- `GSM`

对应本仓库里的文件：

- `dataset/prosqa_train.json`
- `dataset/prosqa_valid.json`
- `dataset/prosqa_test.json`
- `dataset/gsm_train.json`
- `dataset/gsm_valid.json`
- `dataset/gsm_test.json`

### 2.4 训练目标

当前只用标准的 next-step prediction objective：

- 输入前缀 `question + previous steps`
- 预测当前 step `s_i`

这里不引入：

- step-level MTP objective
- token-level MTP objective
- transition model 训练
- rollout 评估

因为这些都不属于实验一。

对于实验二A，当前新增一种目标：

- `decoder_token_mtp`

它仍然只预测当前 step `s_i`，但在 step 的 token 序列内部，对同一个 decoder hidden state 施加多个 future-token 监督。

## 3. 数据接口如何对应实验一与实验二A

`src/mtp_latent/data.py` 当前负责把原始样本统一展开成：

```text
question
s_1
s_2
...
s_n
answer
```

并构造：

```text
x_i = question + s_1 + ... + s_{i-1}
target = s_i
```

对实验一来说，配置里直接把 `max_horizon` 设为 `1`，只做当前 step 的监督。

实验二A也保持 `max_horizon=1`。

原因是：

- 实验二A改变的是 step 内部 token 级别的监督
- 不是 step 级别的多步监督
- 因此仍然只取当前 step `s_i`

因此，从实验一视角看，当前 loader 的要求就是：

1. 能正确读取 `question / steps / answer`
2. 能把每个 step 展开成 prefix-current-step pair
3. 能在 `ProsQA` 和 `GSM` 上一致工作

## 4. 实验二A具体怎么实现

这是当前最重要的实现定义。

### 4.1 预测对象

实验二A不是从 `z_i` 直接预测多个 future step。

它做的是：

- 先按正常方式解码当前 step `s_i`
- 对 decoder 中每个位置的 hidden state
- 不只预测下一个 token
- 还额外预测更远的未来 token

### 4.2 当前实现口径

设当前 step token 序列是：

```text
y_1, y_2, y_3, ..., y_T
```

decoder 在位置 `t` 产生 hidden state `h_t`。

当前代码中：

- `h_t` 的标准头预测 `y_t`
- `h_t` 的 token-horizon-2 头预测 `y_{t+1}`
- `h_t` 的 token-horizon-3 头预测 `y_{t+2}`

也就是说，默认的实验二A实现是：

```text
h_t -> y_t
h_t -> y_{t+1}
h_t -> y_{t+2}
```

### 4.3 预测哪些 token

当前实现默认使用：

- `token_prediction_horizons = [1, 2, 3]`

含义是：

- `1`：标准 next-token supervision
- `2`：额外预测再往后 1 个 token
- `3`：额外预测再往后 2 个 token

### 4.4 用什么 head

当前实现中：

- `horizon=1` 复用 GPT-2 原本的 `lm_head`
- `horizon>1` 使用额外的 token prediction head

所以实验二A不是简单复用同一组 logits 去对不同 target，而是：

- 当前 token 用标准头
- 更远 future token 用专门的额外头

### 4.5 loss 对齐方式

对 `horizon=1`：

```text
logits(h_t) 对齐 target y_t
```

对 `horizon=2`：

```text
logits_2(h_t) 对齐 target y_{t+1}
```

对 `horizon=3`：

```text
logits_3(h_t) 对齐 target y_{t+2}
```

对于超出句长的位置，自动 mask。

### 4.6 当前默认 loss 权重

当前默认配置是：

```yaml
token_prediction_horizons: [1, 2, 3]
token_prediction_weights: [1.0, 0.5, 0.25]
```

即：

- 当前 token loss 权重最高
- 更远 future token loss 逐步减小

### 4.7 它和实验二B的区别

实验二A是：

- token-level 多未来监督
- 仍然只训练当前 step `s_i`

实验二B才是：

- step-level 多未来监督
- 从同一个 `z_i` 预测 `s_i, s_{i+1}, s_{i+2}`

这两者不能混。

## 5. 当前配置应该怎么理解

实验一直接看四个运行条件：

- `ProsQA + NTP-init`
- `ProsQA + MTP-init`
- `GSM + NTP-init`
- `GSM + MTP-init`

每个条件各有一份 YAML。

这些 YAML 的共同原则是：

- 数据路径固定
- 训练目标固定
- 训练超参数固定
- 输出目录独立
- 只有 `model.model_name_or_path` 不同

当前已经补好的四个实验一配置文件是：

- `configs/exp1_prosqa_ntp_init.yaml`
- `configs/exp1_prosqa_mtp_init.yaml`
- `configs/exp1_gsm_ntp_init.yaml`
- `configs/exp1_gsm_mtp_init.yaml`

你只需要修改这些 YAML 中的：

- `model.model_name_or_path`

把它替换成服务器上的实际模型目录即可。

实验二A新增四个配置文件：

- `configs/exp2a_prosqa_ntp_init.yaml`
- `configs/exp2a_prosqa_mtp_init.yaml`
- `configs/exp2a_gsm_ntp_init.yaml`
- `configs/exp2a_gsm_mtp_init.yaml`

它们和实验一相比，主要差别是：

- `codec_objective.name = decoder_token_mtp`
- `model.max_token_mtp_horizon = 3`
- `codec_objective.token_prediction_horizons = [1, 2, 3]`
- `codec_objective.token_prediction_weights = [1.0, 0.5, 0.25]`

## 6. 当前脚本应该产出什么

实验一当前阶段最重要的产出不是复杂评估，而是训练过程曲线。

因此脚本的目标很明确：

1. 启动对应数据集、对应初始化模型的训练
2. 在独立输出目录下保存 checkpoint
3. 产出 TensorBoard 日志

当前最关注的结果是：

- `train loss curve`
- `valid loss curve`
- `primary_loss`
- 实验二A下的 `token_h1_loss / token_h2_loss / token_h3_loss`
- 实验二A下的 `token_h1_acc / token_h2_acc / token_h3_acc`

TensorBoard 日志默认保存在各自实验目录下的：

```text
outputs/<experiment_name>/tensorboard/
```

查看方式：

```bash
tensorboard --logdir outputs
```

## 7. 当前代码和设计文档的对应关系

当前阶段对应设计文档中的：

- 第 5 节：实验一
- 第 6.2 节：实验二A

当前还不作为本轮实现目标的包括：

- step-level MTP objective
- transition model initialization 对照
- continuous vs discretized rollout
- transfer

## 8. 为什么先做实验一和实验二A

因为当前最重要的是先把下面这件事做干净：

> 当模型结构、数据、训练预算完全相同时，仅仅把初始化从 NTP 改成 MTP，训练曲线会不会发生稳定差别？

实验二A是下一步最自然的扩展，因为它仍然只在当前 step 上训练，只是改变 decoder 内部 token 监督方式，不会一下子把 step-level multi-horizon、transition 和 rollout 全部混进来。

## 9. 你接下来如何使用

你的使用方式应该是：

1. 在实验一对应 YAML 中填写服务器上的 NTP / MTP GPT-2 路径
2. 直接运行对应的 `sh` 脚本
3. 训练完成后查看 TensorBoard 的 loss 曲线

当前阶段的成功标准是：

- 四组实验都能一键启动
- 输出目录互不覆盖
- TensorBoard 能直接对比 loss 曲线

只要实验一和实验二A的变量都稳定了，后面再加实验二B和 transition 才是合理的。

## 10. 一键运行脚本

当前已经补好的实验一脚本是：

- `scripts/train_exp1_prosqa_ntp_init.sh`
- `scripts/train_exp1_prosqa_mtp_init.sh`
- `scripts/train_exp1_gsm_ntp_init.sh`
- `scripts/train_exp1_gsm_mtp_init.sh`

TensorBoard 查看脚本：

- `scripts/tensorboard_exp1.sh`
- `scripts/tensorboard_exp2a.sh`

实验二A新增脚本：

- `scripts/train_exp2a_prosqa_ntp_init.sh`
- `scripts/train_exp2a_prosqa_mtp_init.sh`
- `scripts/train_exp2a_gsm_ntp_init.sh`
- `scripts/train_exp2a_gsm_mtp_init.sh`

推荐使用方式：

1. 先修改四个 YAML 中的模型路径
2. 在服务器上直接运行对应脚本
3. 训练结束后用 TensorBoard 查看 `outputs/` 下的曲线

补充说明：

- 当前 CLI 只保留 `train-codec`、`evaluate`、`inspect-data`、`show-history`
- 当前实验一 YAML 不再显式携带 transition 配置块
