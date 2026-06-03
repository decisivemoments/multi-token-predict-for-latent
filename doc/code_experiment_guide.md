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

这里的 `target = s_i` 指的是 step 文本本身。真正送进 decoder 的 token 序列在实现里会改写成：

```text
y_1 y_2 ... y_T [EOS]
```

其中：

- `y_1 ... y_T` 是 `s_i` 的 tokenizer 输出
- `EOS` 用于表示“这个 step 在这里结束”
- tokenizer 本身不再额外自动插入 special tokens，避免和手工加入的 `EOS` 重复
- decoder 不再显式使用 `BOS`；第一个 token 由 latent 直接预测

这一步对实验一和实验二A都是统一的基础接口，不改变两组实验的变量控制。

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

### 3.1 训练时每一步样本到底怎么做

设当前一个训练样本对应：

```text
prefix = x_i
step = s_i
```

记 step 的 token 为：

```text
y_1, y_2, ..., y_T
```

则 decoder 训练序列统一构造成：

```text
target_tokens = [y_1, y_2, ..., y_T, EOS]
```

这里配置里的 `max_step_tokens` 指的是整个 decoder 序列长度上限，也就是：

```text
step tokens + EOS
```

因此真正留给 step 文本本身的最大长度是：

```text
max_step_tokens - 1
```

teacher-forcing 输入不再有人造 `BOS`。decoder 输入由两部分组成：

```text
latent_prefix + [y_1, y_2, ..., y_T]
```

监督目标是：

```text
[y_1, y_2, ..., y_T, EOS]
```

实现里会额外构造一份 `labels`：

- 有效位置保留真实 token id
- padding 位置写成 `-100`

因此 loss 的忽略逻辑不再依赖 `pad_token_id`，而是依赖：

```text
labels == -100
```

这一步的作用是保证：

- 真实 `EOS` 一定参与 loss
- 即使当前 tokenizer 的 `pad_token_id == eos_token_id`
- 也不会把真实 `EOS` 误当成 padding 忽略掉

这意味着：

- latent 直接负责预测第一个 token `y_1`
- 模型不仅学习生成 step 内容
- 还必须学习在合适位置输出 `EOS`
- 之后真正做离散 rollout 时，decoder 才有明确的停止信号

### 3.2 为什么这里必须加 EOS

如果不加 `EOS`，训练目标只能告诉 decoder：

- 当前 token 后面应该接什么内容

但不能告诉 decoder：

- 什么时候这个 step 应该结束

后果是：

- 当前实验一和实验二A的训练 loss 虽然还能算
- 但后面一旦和 transition model 结合，离散解码时就没有可靠的 stop criterion
- 只能靠 `max_step_tokens` 强行截断

这对后续 latent reasoning rollout 不够严谨。

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
y_1, y_2, y_3, ..., y_T, EOS
```

decoder 在位置 `t` 产生 hidden state `h_t`。

当前代码中：

- `h_0` 是 latent prefix 位置，对应预测 `y_1`
- `h_t` 的标准头预测当前位置之后的下一个 token
- `h_t` 的 token-horizon-2 头预测再往后 1 个 token
- `h_t` 的 token-horizon-3 头预测再往后 2 个 token

也就是说，默认的实验二A实现是：

```text
h_0 -> y_1
h_1 -> y_2
...
h_t -> y_{t+1}
h_t -> y_{t+2}
h_t -> y_{t+3}
```

这里要注意，序列尾部的 `EOS` 也属于合法 target。

因此实验二A下，靠近序列结尾的位置会出现这样的训练信号：

```text
h_{T} -> EOS
```

以及如果 horizon 足够大，也可能出现：

```text
h_{T-1} -> EOS
```

这正是我们希望保留的行为，因为 decoder 不仅要会生成文本，还要会判断 step 结束。

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
logits(h_t) 对齐 target y_{t+1}
```

对 `horizon=2`：

```text
logits_2(h_t) 对齐 target y_{t+2}
```

对 `horizon=3`：

```text
logits_3(h_t) 对齐 target y_{t+3}
```

对于超出句长的位置，自动 mask。

在有 `EOS` 后：

- `horizon=1` 的最后一个有效 target 是 `EOS`
- `horizon=2`、`horizon=3` 如果越过 `EOS` 之后没有合法 token，就自然被 mask

因此：

- 实验一的 `token_h1_loss` 现在包含“是否正确结束”
- 实验二A的 `token_h1_loss` 也是同一口径
- 所以实验一和实验二A的 `primary_loss / token_h1_loss` 仍然可以直接比较

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

其中：

- `primary_loss` 统一定义为 `token_h1_loss`
- 它对应“当前 step 的标准 next-token + EOS 预测损失”
- 这是实验一和实验二A之间最适合横向比较的主指标
- 每个 epoch 的 valid 生成样例 JSON

除了 loss 和 TensorBoard 之外，验证阶段还会额外做少量真实生成预览。

当前默认行为是：

- 每个 epoch 在 valid loss 计算结束后
- 额外抽取若干条 valid 样本
- 用当前 codec 从 latent 自回归生成完整 step
- 将 `prefix / gold step / predicted step / 是否正常生成到 EOS` 写入 JSON

默认输出位置：

```text
outputs/<experiment_name>/valid_generations/epoch_001.json
outputs/<experiment_name>/valid_generations/epoch_002.json
...
```

JSON 的作用不是做正式打分，而是帮助你快速看：

- 模型现在生成的 step 大概长什么样
- 是否经常过早结束
- 是否经常生成不出 `EOS`
- 和 gold step 在表面上差多少

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
- transfer

但本轮会把 future transition 需要的 decoder interface 先补完整，也就是：

- step token 训练时显式加入 `EOS`
- 提供真正的 `latent -> autoregressive step generation` 接口
- 为后续 discretized rollout 提前定义统一的 stop rule

### 7.1 后续 generate 接口应该怎么做

当后面需要把 codec 和 transition model 结合时，每一个 latent state `z_i` 不应该只 decode 出 1 个 token，而应该 decode 出完整的一个 reasoning step。

生成流程统一定义为：

1. 给定 `z_i`
2. 不额外喂入 `BOS`
3. 直接由 latent prefix 预测第一个 token
4. 把新 token 追加回输入，继续自回归生成
5. 如果生成到 `EOS`，当前 step 结束
6. 如果一直未生成 `EOS`，则在 `max_step_tokens` 强制停止

也就是说，真正的 step-level generation 终止条件是：

- 首选 `EOS`
- 其次是 `max_step_tokens` 兜底截断

### 7.2 后续 discretized rollout 每一步怎么做

未来若做离散 latent rollout，每一步统一按下面的过程：

1. 从当前 prefix 编码得到 latent：

```text
z_i = Encoder(question + s_1 + ... + s_{i-1})
```

2. transition 预测下一 latent：

```text
\hat{z}_{i+1} = Transition(z_i)
```

3. 用 decoder 从 `\hat{z}_{i+1}` 自回归生成完整 step：

```text
\hat{s}_{i+1} = Decoder.generate_step(\hat{z}_{i+1})
```

其中生成从 latent 直接开始，到 `EOS` 结束。

4. 把生成出的 `\hat{s}_{i+1}` 拼回 prefix：

```text
question + s_1 + ... + s_{i-1} + \hat{s}_{i+1}
```

5. 重新编码得到新的离散化 latent：

```text
z'_{i+1} = Encoder(question + s_1 + ... + s_{i-1} + \hat{s}_{i+1})
```

6. 再进入下一步 rollout

这个接口和当前实验一、实验二A训练目标是一致的，因为训练时 decoder 已经学会：

- 如何从 latent 直接开始生成一个 step
- 如何在适当位置输出 `EOS`

因此，后续 transition 接进来时，不需要再改 codec 训练口径。

### 7.3 当前 codec 还缺 answer supervision

当前 codec 只覆盖了 reasoning step 的生成监督：

- `question -> s_1`
- `question + s_1 -> s_2`
- ...
- `question + s_1 + ... + s_{n-1} -> s_n`

但完整题目还差最后一步：

```text
question + s_1 + ... + s_n -> answer
```

因此如果后面要把整条题的生成链条做完整，codec 也必须把 `answer` 纳入 target。

统一后的完整 trace 应该写成：

```text
question
s_1
s_2
...
s_n
answer
```

这样 codec 最后一个 prefix-target pair 就是：

```text
prefix = question + s_1 + ... + s_n
target = answer
```

后面 transition model 的设计也会沿用同样的“最后一个目标是 answer”的定义。

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

## 10. Transition Model 设计

下面这部分先固定实现口径，暂时还不展开实验脚本。

### 10.1 transition model 要解决什么问题

codec 训练完成并冻结之后，每个 gold reasoning trace 都可以得到一串 prefix latent：

```text
z_1 = Encoder(question)
z_2 = Encoder(question + s_1)
z_3 = Encoder(question + s_1 + s_2)
...
z_k = Encoder(question + s_1 + ... + s_{k-1})
```

这里 `z_i` 的语义是：

- 它表示“看到 question 和前 `i-1` 个 step 之后的状态”
- 它对应的下一个 gold step 是 `s_i`

因此 transition model 的目标不是单纯做：

```text
z_i -> z_{i+1}
```

而是学一个更强的、可 rollout 的 causal predictor：

- 先从 question token 序列推出 `s_1`
- 再从 `question + z_1` 推出 `s_2`
- 再从 `question + z_1 + z_2` 推出 `s_3`
- 以此类推

如果把整条题做完整，还需要再补一项：

- 从最后一个 reasoning latent 推出 `answer`

### 10.2 训练输入如何组织

给定一个样本：

```text
question = q_1, q_2, ..., q_n
steps = s_1, s_2, ..., s_m
```

先用冻结的 codec encoder 构造：

```text
z_1 = Encoder(question)
z_2 = Encoder(question + s_1)
...
z_{m-1} = Encoder(question + s_1 + ... + s_{m-2})
z_m = Encoder(question + s_1 + ... + s_{m-1})
```

然后 transition model 的一条训练序列定义成：

```text
[q_1, q_2, ..., q_n, z_1, z_2, ..., z_m]
```

也就是说：

- question 部分用离散 token
- reasoning history 部分用连续 latent token

这是一个 mixed sequence。

### 10.3 哪些位置有监督

不是整条序列每个位置都要监督。

当前定义是：

- `q_1 ... q_{n-1}` 没有 step supervision
- `q_n` 的 last hidden state 负责预测 `s_1`
- `z_1` 的 last hidden state 负责预测 `s_2`
- `z_2` 的 last hidden state 负责预测 `s_3`
- ...
- `z_{m-1}` 的 last hidden state 负责预测 `s_m`
- `z_m` 的 last hidden state 负责预测 `answer`

因此监督位置集合是：

```text
[q_n, z_1, z_2, ..., z_m]
```

监督目标集合是：

```text
[s_1, s_2, s_3, ..., s_m, answer]
```

这和 latent 的定义是严格对齐的：

- `q_n` 看到了完整 question，所以它对应“第一个要生成的 step”
- `z_1` 表示看到 question 之后的第一个 latent state，因此它对应“第二个要生成的 step”
- `z_i` 表示看到前 `i-1` 个 gold steps 之后的状态，因此它对应 `s_{i+1}`
- `z_m` 表示已经看到全部 reasoning steps，因此它对应最终 `answer`

### 10.4 transition backbone 具体长什么样

第一版实现直接使用一个 GPT-2 style causal transformer 作为 transition backbone，并且它的参数初始化来自 `model.model_name_or_path` 指向的预训练 GPT-2 权重，而不是随机初始化。

它的输入嵌入由两部分组成：

1. question token embedding
2. latent token embedding

更具体地说：

- 对 `q_1 ... q_n`，使用 transition backbone 自己的 token embedding
- 对 `z_i`，先用一个线性层把 `latent_dim` 投到 backbone hidden size，再把它当成一个连续 token embedding 喂进去

因此需要两个附加投影：

```text
latent_in_proj: latent_dim -> transition_hidden_dim
latent_out_proj: transition_hidden_dim -> latent_dim
```

其中：

- `latent_in_proj` 用于把 codec latent 放进 transition transformer
- `latent_out_proj` 用于把 transition 某个位置的 hidden state 再投回 decoder 可接受的 latent 空间

### 10.5 监督信号怎么施加

这是这一部分最关键的实现细节。

设 transition backbone 在监督位置产生 hidden state：

```text
h(q_n), h(z_1), h(z_2), ..., h(z_m)
```

先经过：

```text
u_1 = latent_out_proj(h(q_n))
u_2 = latent_out_proj(h(z_1))
u_3 = latent_out_proj(h(z_2))
...
u_m = latent_out_proj(h(z_{m-1}))
u_{m+1} = latent_out_proj(h(z_m))
```

然后把这些 `u_i` 当成“预测得到的 latent state”，交给**冻结的 codec decoder** 去 decode。

也就是：

```text
Decoder(u_1) -> s_1
Decoder(u_2) -> s_2
...
Decoder(u_m) -> s_m
Decoder(u_{m+1}) -> answer
```

训练 loss 不是直接对 hidden state 做 MSE，而是：

- 用 codec decoder 对每个 `u_i` 做 teacher-forcing step generation
- 对应 gold step token 序列算 cross-entropy

因此 transition model 的主监督是：

```text
transition hidden state
-> latent_out_proj
-> frozen decoder
-> token CE loss on gold step
```

第一版实现里，这个 decode loss 就作为 transition 的主 loss。

### 10.6 trace-level completion 应该怎么做

这里要明确区分两种“结束”：

1. `step-level EOS`
- 它只表示当前一条文本结束
- 可能是一条 reasoning step 结束
- 也可能是 final answer 文本结束

2. `trace-level completion`
- 它表示整道题的 reasoning trace 是否已经走到“该输出 final answer”的阶段
- 它不是 decoder 文本 token
- 它是 rollout 控制信号

第一版建议把 trace-level completion 做成一个显式的二分类头：

```text
next_type_head(h) -> {step, answer}
```

也就是对每个监督位置，同时预测：

1. 下一个要 decode 的对象类型
2. 下一个要 decode 的文本内容

具体标签定义如下：

```text
q_n      -> next_type = step,   next_text = s_1
z_1      -> next_type = step,   next_text = s_2
...
z_{m-1}  -> next_type = step,   next_text = s_m
z_m      -> next_type = answer, next_text = answer
```

这个定义的关键点是：

- trace-level completion 不表示“现在什么都不生成，直接停掉”
- 它表示“下一次 decode 的目标类型从 reasoning step 切换成 final answer”
- 一旦 `next_type = answer`，系统就用 decoder 生成 answer 文本
- answer 文本生成到 step-level `EOS` 后，整条题结束

### 10.7 transition 的训练 loss

基于上面的定义，transition 每个监督位置都有两类损失：

1. `type loss`

```text
L_type = CE(next_type_logits, next_type_label)
```

2. `decode loss`

- 如果 `next_type_label = step`，就 decode 到对应 `s_i`
- 如果 `next_type_label = answer`，就 decode 到对应 `answer`

也就是：

```text
L_decode = frozen_decoder_CE(predicted_latent_like_state, target_text)
```

第一版总 loss 建议直接写成：

```text
L = L_type + L_decode
```

### 10.8 为什么第一版先不用 latent regression

还有一种更传统的做法是：

```text
T(z_i) -> z_{i+1}
```

然后直接对 `z_{i+1}` 做回归或相似度 loss。

这条路当然也可以做，但第一版不作为主实现，原因是：

1. 你的需求里更强调“某个位置的 hidden state 是否真的能解码成正确下一 step”
2. decoder-based supervision 更接近最终我们关心的可读 reasoning state
3. 这样更容易直接检查 transition hidden state 是否仍在 decoder-readable manifold 上

后面如果要增强训练稳定性，可以再加辅助项：

- latent regression loss
- contrastive retrieval loss

但第一版先不混进来。

### 10.9 一个完整例子

设样本是：

```text
question: "Tom bought his games for $200. They tripled in value and he then sold 40% of them. How much did he sell the games for?"

s_1 = "<<200*3=600>>"
s_2 = "<<600*.4=240>>"
answer = "240"
```

那么：

```text
z_1 = Encoder(question)
z_2 = Encoder(question + s_1)
```

因为这里只有两个 steps，所以 transition 训练序列是：

```text
[q_1, q_2, ..., q_n, z_1, z_2]
```

监督位置和目标是：

```text
q_n  -> type=step,   target=s_1
z_1  -> type=step,   target=s_2
z_2  -> type=answer, target=answer
```

具体 forward 过程是：

1. 把 `question` token embedding 输入 transition backbone
2. 把 `z_1, z_2` 经过 `latent_in_proj` 作为连续 token 接到 question 后面
3. 用 causal attention 跑完整条序列
4. 取 `q_n` 位置 hidden state，投影成 `u_1`
5. 取 `z_1` 位置 hidden state，投影成 `u_2`
6. 取 `z_2` 位置 hidden state，投影成 `u_3`
7. 用 `next_type_head` 分类：

```text
q_n -> step
z_1 -> step
z_2 -> answer
```

8. 用冻结 decoder：

```text
Decoder(u_1) 监督到 "<<200*3=600>>"
Decoder(u_2) 监督到 "<<600*.4=240>>"
Decoder(u_3) 监督到 "240"
```

9. 三个 decode CE loss 加上三个 type CE loss，求和或求平均，作为这条样本的 transition loss

### 10.10 训练时使用 gold latent，rollout 时使用 predicted latent

transition 训练阶段先采用 teacher-forcing 口径：

- question token 用 gold
- latent token `z_i` 也用 gold codec encoder 提取结果

也就是说，训练时输入的是：

```text
[q_1, ..., q_n, z_1^{gold}, z_2^{gold}, ...]
```

这样先保证 transition backbone 学会在“正确历史”条件下预测下一 step。

到了 rollout / inference 阶段，再切成：

- 第一个监督位置先用 `next_type_head` 判断下一项是 `step` 还是 `answer`
- 如果是 `step`，就 decode 出 `\hat{s}_1`
- 再把 `\hat{s}_1` 重新 encode 成 `\hat{z}_1`
- 再把 `\hat{z}_1` 接回 sequence，继续预测下一项
- 如果某一步 `next_type_head` 判成 `answer`，就 decode 出 `\hat{answer}`
- `\hat{answer}` 生成到 step-level `EOS` 后，整条题结束

所以：

- 训练是 gold latent teacher forcing
- 推理是 predicted latent autoregressive rollout

第一版实现先把训练接口做稳定，rollout 作为下一步单独接。

### 10.11 当前这版 transition 实现边界

为了先把接口做干净，第一版 transition 建议保持下面这些限制：

1. codec encoder 和 codec decoder 冻结
2. transition model 单独训练
3. 主 loss 用 `type CE + decoder CE`
4. 不先混入 latent MSE / cosine / contrastive 辅助 loss
5. 不先做 multi-step scheduled sampling

这样我们先回答一个最干净的问题：

> 不同 codec 产生的 latent state，是否会影响 transition backbone 学出可 decode 的下一 step state？

以及：

> transition backbone 是否能学会在合适的时候把“下一项类型”从 reasoning step 切换成 final answer？

当前实现中的初始化口径固定为：

- `transition.codec_checkpoint`：加载实验一或实验二A训练完成后保存的 codec `pt`，恢复 frozen encoder-decoder
- `model.model_name_or_path`：加载预训练 GPT-2，作为 transition backbone 的初始化
- `transition.init_checkpoint`：只保留给“继续训练已有 transition checkpoint”的场景，不用于第一轮实验初始化

当前实现中的 transition 数据准备口径固定为：

- dataloader 初始化阶段不再预先把全量 prefix 过一遍 codec encoder
- dataset 只保存 `question_text`、每个 latent 对应的 `prefix_text`、以及对应的 target 文本
- 训练和验证时，在每个 batch 内部用 frozen codec encoder 现算该 batch 的 `z_1 ... z_m`

这样训练启动后会立刻进入 epoch 和 step 日志，不会先卡在一段长时间的全量 latent 预处理上。

等这部分跑通之后，再逐步加：

- rollout evaluation
- latent regression 辅助项
- experiment 3 / 4 的完整脚本

## 11. 一键运行脚本

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

transition 训练脚本：

- `scripts/train_exp3_transition_exp1_prosqa_ntp.sh`
- `scripts/train_exp3_transition_exp1_prosqa_mtp.sh`
- `scripts/train_exp3_transition_exp1_gsm_ntp.sh`
- `scripts/train_exp3_transition_exp1_gsm_mtp.sh`
- `scripts/train_exp3_transition_exp2a_prosqa_ntp.sh`
- `scripts/train_exp3_transition_exp2a_prosqa_mtp.sh`
- `scripts/train_exp3_transition_exp2a_gsm_ntp.sh`
- `scripts/train_exp3_transition_exp2a_gsm_mtp.sh`

推荐使用方式：

1. 先修改 YAML 中的 `model.tokenizer_name_or_path` 和 `model.model_name_or_path`
2. 确认 `transition.codec_checkpoint` 指向实验一或实验二A训练产出的 `codec_best.pt`
3. 在服务器上直接运行对应脚本
4. 训练结束后用 TensorBoard 查看 `outputs/` 下的曲线

补充说明：

- 当前 CLI 现在支持 `train-codec`、`train-transition`、`evaluate`、`inspect-data`、`show-history`
- transition 配置需要显式填写 `transition.codec_checkpoint`
- 当前 exp3 YAML 已经清理为最小有效集：codec objective 和随机初始化相关字段不再保留在 exp3 配置里

transition valid 阶段当前除了 loss 外，还会额外计算三类 answer 指标：

- `teacher_forced_answer_acc`
  - 给出完整 gold steps，只检查最后一个 answer 位置的 predicted latent 能不能 decode 出正确 answer
- `rollout_direct_answer_acc`
  - 只给 question，让 transition 自己 rollout
  - 每一步直接把上一时刻 transition 预测出的 latent 继续送回 transition
- `rollout_reencode_answer_acc`
  - 只给 question，让 transition 自己 rollout
  - 每一步先 decode 成文本 step，再把 `question + generated steps` 重新交给 codec encoder 编成 latent，再送回 transition

同时还会记录：

- `rollout_direct_answer_stop_rate`
- `rollout_reencode_answer_stop_rate`

它们表示 rollout 过程中，模型有多少比例真的把 `next_type` 切换成了 `answer`，而不是一直停留在 `step`。

另外，codec 和 transition 训练当前都会在各自 `output_dir` 下额外保存一个紧凑版 valid 摘要 JSON：

- `codec_valid_compact.json`
- `transition_valid_compact.json`

这两个文件每个 epoch 只保留少量关键指标，目的是方便直接复制文本给外部模型做结果分析，而不必先把 TensorBoard 曲线手工转述一遍。

另外，codec 当前还新增了两项 answer 专属 valid 指标：

- `answer_token_loss`
- `answer_token_acc`

它们只在 `target_kind == answer` 的样本上统计，用来判断：

- answer token 本身是否已经学会
- 还是 answer 的 free-generation exact match 指标过于严格

为了缓解 answer 样本在一条 trace 里只出现一次的问题，codec 训练还支持：

- `codec_objective.answer_loss_weight`

当该值大于 `1.0` 时，训练会对 `answer` 样本分配更高的 loss 权重。
