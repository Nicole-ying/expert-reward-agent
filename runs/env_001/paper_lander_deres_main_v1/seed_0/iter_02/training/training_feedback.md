# Training Feedback

## Training outcome
score=124.319811, len=873.900000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| approach_quality_reward | 0.642615 | 0.642615 | 1.000000 | 0.642615 | 280.547038 |
| attitude_penalty | -0.000478 | 0.000478 | 1.000000 | -0.000478 | -0.208650 |
| progress_reward | 0.001944 | 0.002291 | 0.999817 | 0.001944 | 0.848490 |
| total_reward | 0.194250 | 0.194294 | 1.000000 | 0.194250 | 84.803951 |
| original_env_reward | 0.039010 | 2.371493 | 1.000000 | 0.039010 | 17.030718 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_quality_reward | 393.841891 | 393.841891 | 1.710179 | 852.017275 | 1635 |
| attitude_penalty | -0.292658 | 0.292658 | -2.857381 | -0.007491 | 1635 |
| progress_reward | 1.190195 | 1.257791 | -9.179923 | 1.420406 | 1635 |
| total_reward | 119.050104 | 119.053141 | -1.802502 | 256.684455 | 1635 |

## Distribution
- score: mean=124.319811, min=-11.278966, max=183.199756
- episode_length: mean=873.900000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
