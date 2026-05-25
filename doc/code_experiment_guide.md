# 代码实验说明

本文档只对齐当前阶段真正要做的事情：

> 先把设计文档中的“实验一：MTP initialization 是否让 codec 更好学”实现清楚、变量控制清楚、运行入口清楚。

这里不展开实验二、实验三、rollout 或 transfer。当前代码和配置都应优先服务于实验一。

## 1. 当前只验证什么

当前只做下面这个对照：

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

## 3. 数据接口如何对应实验一

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

因此，从实验一视角看，当前 loader 的要求就是：

1. 能正确读取 `question / steps / answer`
2. 能把每个 step 展开成 prefix-current-step pair
3. 能在 `ProsQA` 和 `GSM` 上一致工作

## 4. 当前配置应该怎么理解

实验一不再强调一堆抽象配置名字，而是直接看四个运行条件：

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

## 5. 当前脚本应该产出什么

实验一当前阶段最重要的产出不是复杂评估，而是训练过程曲线。

因此脚本的目标很明确：

1. 启动对应数据集、对应初始化模型的训练
2. 在独立输出目录下保存 checkpoint
3. 产出 TensorBoard 日志

当前最关注的结果是：

- `train loss curve`
- `valid loss curve`

TensorBoard 日志默认保存在各自实验目录下的：

```text
outputs/<experiment_name>/tensorboard/
```

查看方式：

```bash
tensorboard --logdir outputs
```

## 6. 当前代码和设计文档的对应关系

当前阶段只对应设计文档中的第 5 节，也就是实验一：

- 比较 `Codec-NTP-init`
- 比较 `Codec-MTP-init`
- 保持 objective、数据、预算一致
- 观察训练和验证 loss 曲线

设计文档里提到的这些更后续的东西，当前都不作为本轮实现目标：

- step-level MTP objective
- transition model initialization 对照
- continuous vs discretized rollout
- transfer

## 7. 为什么现在要先收敛到实验一

因为当前最重要的是先把下面这件事做干净：

> 当模型结构、数据、训练预算完全相同时，仅仅把初始化从 NTP 改成 MTP，训练曲线会不会发生稳定差别？

如果这个问题都没有被严格控制并跑清楚，后面再叠加 objective、transition 和 rollout，只会重新把变量混在一起。

## 8. 你接下来如何使用

你的使用方式应该是：

1. 在实验一对应 YAML 中填写服务器上的 NTP / MTP GPT-2 路径
2. 直接运行对应的 `sh` 脚本
3. 训练完成后查看 TensorBoard 的 loss 曲线

当前阶段的成功标准也很简单：

- 四组实验都能一键启动
- 输出目录互不覆盖
- TensorBoard 能直接对比 loss 曲线

只要这一步稳定了，后面再继续加实验二和 transition 相关逻辑才是合理的。

## 9. 一键运行脚本

当前已经补好的实验一脚本是：

- `scripts/train_exp1_prosqa_ntp_init.sh`
- `scripts/train_exp1_prosqa_mtp_init.sh`
- `scripts/train_exp1_gsm_ntp_init.sh`
- `scripts/train_exp1_gsm_mtp_init.sh`

TensorBoard 查看脚本：

- `scripts/tensorboard_exp1.sh`

推荐使用方式：

1. 先修改四个 YAML 中的模型路径
2. 在服务器上直接运行对应脚本
3. 训练结束后用 TensorBoard 查看 `outputs/` 下的曲线

补充说明：

- 当前 CLI 只保留 `train-codec`、`evaluate`、`inspect-data`、`show-history`
- 当前实验一 YAML 不再显式携带 transition 配置块
