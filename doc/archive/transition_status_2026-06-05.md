# Transition 实验阶段性状态（历史记录）

> 归档说明：这份文档记录的是 2026-06-05 前后的 transition-first 思路。当前主线已经调整为先做 representation usefulness map，再决定是否继续强化 transition。当前状态请看 `doc/mtp_latent_reasoning_experiment_design.md` 和 `doc/representation_usefulness_analysis_plan.md`。

## 1. 当前现象

我们当前关注的是 transition model 在 `direct rollout` 下的表现，也就是：

- 只给问题
- transition 自己逐步 rollout 后续 latent
- 再由 codec decoder 解码 step / answer

目前的初步观察是：

- `mtp init` 比 `ntp init` 更好
- 但 direct rollout 的正确率绝对值仍然很低，大约在 `0.05` 左右

这个结果说明两件事：

- 初始化方向是有信号的，`mtp init` 更适合作为 transition 的起点
- 但当前 transition 的实际可用性仍然不够，`0.05` 的正确率太低，不能满足我们的目标

所以这一阶段的核心问题不是“哪种 init 更好”，而是：

> 怎么进一步提升 transition model 的性能

## 2. 连续表征空间优化的尝试

我们也尝试对 transition 的 latent 辅助监督做了一些调整，例如：

- `cosine`
- `infonce`
- `infonce_huber`

这里的核心假设是：

> 如果 transition 预测的 latent 能更好地贴近目标 encoder latent，那么它所在的连续表征空间可能会更容易优化。

目前看到的结果是：

- latent 相关指标在训练前几轮会下降
- 但通常两三轮后就进入平台期

所以 loss 调整目前更像是提供了局部改进，并没有根本改变 direct rollout 表现。这部分先不作为现阶段重点。

## 3. 当前判断

我们现在更倾向于认为，当前更需要回答的是：

> codec latent 到底是不是一个适合 transition 建模的状态空间

## 4. 下一步：做表征分析

下一步我们希望先做表征层面的诊断，而不是继续优先调 loss。

### A. Gold latent 几何结构分析

目标：

- 看同一道题中的 `z1, z2, z3, ...` 是否真的构成稳定的 latent 轨迹

计划看：

- 相邻 step latent 的相似度
- 非相邻 step latent 的相似度
- 不同题随机 latent 的相似度

想回答的问题是：

- 相邻 step 是否真的比随机 latent 更接近
- latent 序列本身是否有局部连续结构

### B. Decoder 对 latent 偏差的敏感性分析

目标：

- 看 decoder 对 latent 偏差到底有多敏感

计划做：

- 从 gold latent 出发加噪声或做插值扰动
- 观察 decode 结果什么时候明显崩掉

想回答的问题是：

- latent 相似到什么程度，decoder 才真正可用
- 当前学到的 latent 对齐是否本身就还远远不够

### C. 重新考虑 transition 的目标形式

如果表征分析表明“绝对 latent 回归”不自然，下一步会考虑：

- 预测 `delta latent`
- 或者做 residual update

也就是不直接预测下一个绝对 latent，而是预测状态更新量。

## 5. 总结

1. `mtp init` 比 `ntp init` 更好，这是一个正信号。  
2. 但 direct rollout 正确率仍然只有 `0.05` 左右，绝对表现太低，transition 性能仍需显著提升。  
3. 当前下一步不再把重点放在调 loss，而是优先做表征分析，先判断 latent 空间本身是否适合作为 transition state。  
