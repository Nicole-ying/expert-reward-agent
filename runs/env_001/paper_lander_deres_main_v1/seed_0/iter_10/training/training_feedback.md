# Training Feedback

## Training outcome
score=-108.009874, len=68.450000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| attitude_penalty | -0.001368 | 0.001368 | 1.000000 | -0.001368 | -0.079919 |
| progress_reward | 0.016195 | 0.017121 | 0.999981 | 0.016196 | 0.945938 |
| total_reward | 0.014827 | 0.017014 | 1.000000 | 0.014827 | 0.866018 |
| original_env_reward | -1.573843 | 2.398590 | 1.000000 | -1.573843 | -91.924430 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_penalty | -0.095606 | 0.095606 | -8.062759 | -0.005352 | 14362 |
| progress_reward | 1.131561 | 1.131583 | -0.157786 | 1.412644 | 14362 |
| total_reward | 1.035955 | 1.094970 | -7.345126 | 1.364735 | 14362 |

## Distribution
- score: mean=-108.009874, min=-124.510568, max=-89.013173
- episode_length: mean=68.450000
- early_terminal (<150 steps + score<-50): 20/20 (100%)
- errors: 0
