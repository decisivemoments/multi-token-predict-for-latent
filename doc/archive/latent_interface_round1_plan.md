# Latent 接口第一轮计划（历史记录）

> 归档说明：这份文档记录的是 transition 接口第一轮工程调整。当前研究主线不再默认 transition 是下一步，而是先验证 codec latent 的判别性、verifier 能力和轨迹几何。当前状态请看 `doc/mtp_latent_reasoning_experiment_design.md`。

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

## transition 接口当前结论

在继续观察 transition 训练之后，我们把接口形式先收回到更保守、也更容易验证的一版：

1. `latent_in_proj` 不应该随机初始化  
   因为当前已经统一 `latent_dim = hidden_dim = 768`，更自然的做法是：
   - `latent_in_proj` identity init
   - 让输入 latent token 在训练初期尽量无扭曲地进入 transition backbone

2. `latent_out_proj` 先使用 plain projection  
   当前先不再假设：
   - `predicted_latent = h + delta(h)`

   而是回到更保守的：
   - `predicted_latent = latent_out_proj(h)`

3. `latent_out_proj` 先使用 identity init  
   这样初始时：
   - `latent_out_proj(h) ≈ h`

   这个版本保留了“输出先接近恒等映射”的好处，但不会像 hidden-state residual 那样把模型结构绑定在一个过强的先验上。

这样初始时：

- `latent_in_proj(z) ≈ z`
- `predicted_latent ≈ h`

当前这版的核心想法是：

> 先让输出接口从近似恒等映射开始，再由训练自己决定如何把 transition hidden state 映射到 codec latent manifold。

## 当前不包含的内容

这些放到下一轮再做：

1. transition 接口对照实验  
   包括：
   - direct latent passing
   - hidden-state residual vs plain projection
   - input-latent residual vs hidden-state residual

2. 更宽松的 answer 评估  
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

如果 plain projection + identity init 之后，transition 仍然明显下不去，下一轮最自然的实验就是做更明确的接口对照：

- plain projection：`predicted_latent = W h`
- hidden-state residual：`predicted_latent = h + W h`
- input-latent residual：`predicted_latent = z_i + W h(z_i)`

这样可以直接比较：

> transition 更像是在“修正当前 hidden state”，还是在“更新已有 latent 状态”。
