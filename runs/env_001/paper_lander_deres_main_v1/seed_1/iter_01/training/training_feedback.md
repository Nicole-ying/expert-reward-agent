# Training Feedback

## Training outcome
score=241.414498, len=394.000000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| landing_quality_bonus | 0.369599 | 0.369599 | 0.541875 | 0.682074 | 89.453967 |
| progress_reward | 0.003847 | 0.004132 | 0.994395 | 0.003868 | 0.931035 |
| total_reward | 0.373445 | 0.373656 | 0.999976 | 0.373454 | 90.385002 |
| original_env_reward | -0.128733 | 1.590839 | 1.000000 | -0.128733 | -31.157284 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_quality_bonus | 121.322329 | 121.322329 | 0.000000 | 817.357410 | 3047 |
| progress_reward | 1.265536 | 1.265536 | 0.329691 | 1.420877 | 3047 |
| total_reward | 122.587865 | 122.587865 | 0.329691 | 818.757685 | 3047 |

## Distribution
- score: mean=241.414498, min=81.923051, max=298.037395
- episode_length: mean=394.000000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
