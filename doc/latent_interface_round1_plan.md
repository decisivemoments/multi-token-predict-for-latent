# Latent 接口第一轮计划

## 背景

当前实验里我们观察到两个相关问题：

1. codec 和 transition 之间的 latent 接口可能本身就过于难学。
2. ProsQA 上的训练对学习率比较敏感，而且最终 answer 的生成效果明显偏弱。

此前的一个关键结构问题是：

- GPT-2 的 hidden size 是 `768`
- 旧版本 latent 接口使用的是 `latent_dim = 256`

这相当于额外引入了一个压缩瓶颈，但这个瓶颈并不是我们当前实验真正想研究的变量。

## 当前结论

对于当前这组实验，更合理的做法是把 latent 空间和预训练模型的 hidden size 对齐：

- `latent_dim = 768`
- codec encoder hidden size = `768`
- codec decoder embedding size = `768`
- transition backbone hidden size = `768`

这样可以减少额外的信息损失，也让下面这些接口更自然：

- codec encoder -> latent
- latent -> codec decoder
- codec latent -> transition backbone
- transition backbone -> codec decoder

## 核心判断

当前 transition 准确率低，不一定主要是 reasoning 能力不足，更可能有一部分来自 latent 接口本身不稳定。

目前最可疑的两个来源是：

1. latent 压缩瓶颈破坏了 decoder-readable manifold。
2. projection 层随机初始化，打乱了预训练 GPT-2 的特征空间。

## 第一轮范围

第一轮只改初始化和优化，不改 transition 的核心结构。

### 1. latent 维度对齐

所有当前实验统一使用：

- `latent_dim = 768`

目的：

- 不把“压缩表示”问题混入当前实验
- 让 latent 的尺度和 GPT-2 hidden states 一致
- 为下一轮 residual transition 做准备

### 2. codec projection 做 identity init

对 codec 的两个 projection：

- `latent_proj`
- `decoder_latent_proj`

如果输入输出维度一致，则初始化成 identity：

- weight = 单位矩阵
- bias = 0

预期作用：

- encoder 输出在训练初期可以无扭曲地进入 latent 空间
- latent 进入 decoder 条件通路时，不会先经历一次随机映射
- codec 训练的起点会更稳定

### 3. 学习率调度器

增加一个简单稳定的调度器：

- linear warmup
- cosine decay

预期作用：

- 稳定训练初期 projection 的适配过程
- 减少 ProsQA 上中后期震荡
- 让 25 epoch 训练比固定学习率更有效

## 第一轮新增诊断

在 ProsQA 上，目前 `answer_acc` 提升了，但仍然偏低。这里不能只看 answer 的 free-generation exact match，需要把诊断再拆开。

因此第一轮额外加入两件事：

### 1. answer 专属 valid 指标

valid 阶段单独统计：

- `answer_token_loss`
- `answer_token_acc`

它们只在 `target_kind == answer` 的样本上计算，用来回答：

- answer token 本身是否已经学会
- 还是 token 其实已经学会，但 exact match / 自回归生成仍然偏弱

### 2. answer-weighted codec loss

训练时，对 `answer` 样本增加 loss 权重：

- `answer_loss_weight > 1`

目的：

- 缓解每条 trace 里 answer 监督只出现一次、天然样本占比偏低的问题
- 让模型在训练中更重视最终答案生成

## 当前不包含的内容

这些放到下一轮再做：

1. transition 的 residual latent update  
   在 latent 位置使用：
   `predicted_latent = input_latent + delta`

2. transition 接口对照实验  
   包括：
   - direct latent passing
   - residual vs non-residual

3. 更宽松的 answer 评估  
   例如：
   - numeric answer match
   - value-only correctness

## 新增 sanity check：SFT 基线

为了判断 ProsQA 这个任务是不是连普通 SFT 都难以学会，当前补充三种 SFT 基线：

1. `next_step_sft`
   - 输入：`question + previous_steps`
   - 输出：`current_step`
   - 作用：检查不用 latent 压缩时，模型能否直接学会下一步推理生成

2. `answer_from_steps_sft`
   - 输入：`question + all_steps`
   - 输出：`answer`
   - 作用：检查如果把完整推理轨迹都给模型，它能否稳定生成最终答案

3. `answer_from_question_sft`
   - 输入：`question`
   - 输出：`answer`
   - 作用：检查不显式提供中间步骤时，模型在最终问答上的直接上限

这三组 baseline 的意义是：

- 如果普通 SFT 都学不会，那么当前 latent codec 方案更难，结果偏低就不奇怪
- 如果普通 SFT 能学会，而 latent codec 学不会，就说明主要瓶颈在 latent 接口和条件化方式

## 下一轮方向

如果第一轮后 codec 明显改善，但 transition 仍然弱，下一轮最自然的结构改动是：

- `q_n -> latent` 继续使用普通投影
- `z_i -> z_{i+1}` 的 latent 位置改成 residual update

这样可以更直接地测试：

> transition 更适合被建模成“latent 状态更新器”，还是“从零生成下一个 latent 的模型”。
