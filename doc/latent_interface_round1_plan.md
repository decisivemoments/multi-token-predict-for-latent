# Latent Interface Round 1 Plan

## Background

We observed two related issues:

1. The codec and transition interface may be unnecessarily hard to learn.
2. ProsQA training is sensitive to learning rate and often converges poorly.

At the same time, the pretrained GPT-2 hidden size is `768`, while the old latent interface used a smaller `latent_dim=256`. That introduces an extra compression bottleneck which is not part of the core experimental question.

## Current conclusion

For the current experiment family, we align the latent space with the pretrained model hidden size:

- `latent_dim = 768`
- codec encoder hidden size = `768`
- codec decoder embedding size = `768`
- transition backbone hidden size = `768`

This reduces unnecessary information loss and makes the interface between:

- codec encoder -> latent
- latent -> codec decoder
- codec latent -> transition backbone
- transition backbone -> codec decoder

much easier to reason about.

## Main hypothesis

The current low transition accuracy is likely not only a reasoning problem. A significant part of the error may come from a fragile latent interface.

Two likely causes:

1. A mismatched or compressed latent bottleneck makes decoder-readable states hard to preserve.
2. Randomly initialized projection layers disrupt the pretrained GPT-2 feature space at the start of training.

## Round 1 scope

Round 1 only changes the initialization and optimization side. It does not yet change the transition structure itself.

### 1. Latent dimension alignment

Use `latent_dim=768` across current experiment configs.

Rationale:

- avoid adding a compression problem into the experiment
- keep latent states in the same scale and dimensionality as GPT-2 hidden states
- make later residual transition experiments more natural

### 2. Identity initialization for codec projections

For the codec:

- `latent_proj`
- `decoder_latent_proj`

if input and output dimensions match, initialize them as identity:

- weight = identity matrix
- bias = zero

Expected effect:

- encoder output can pass into latent space without distortion at initialization
- latent can enter decoder conditioning path without an unnecessary random transformation
- codec training should start from a much better operating point

### 3. Learning-rate scheduler

Add a simple training scheduler:

- linear warmup
- cosine decay

Expected effect:

- stabilize the early phase where the bridge layers start adapting
- reduce late-stage oscillation on ProsQA
- make 25-epoch training more effective than a fixed LR

## Not included in Round 1

These are postponed to the next round:

1. Residual transition latent update
   - for latent positions, use `predicted_latent = input_latent + delta`
2. Transition interface ablations
   - direct latent passing
   - residual vs non-residual comparison
3. Structured JSON metric snapshots for easier external analysis

## Next round idea

After Round 1, if codec training improves but transition is still weak, the next structural change should be:

- keep `q_n -> latent` as a normal projection
- use residual latent update on latent-token positions in transition

This is the cleanest way to test whether transition should behave as a latent state update model rather than a full latent generator.
