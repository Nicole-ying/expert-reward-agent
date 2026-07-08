# Training Feedback

## Training outcome
score=139.033146, len=1000.000000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| approach_quality_reward | 0.063388 | 0.063388 | 1.000000 | 0.063388 | 24.433713 |
| attitude_penalty | -0.000453 | 0.000453 | 1.000000 | -0.000453 | -0.174770 |
| progress_reward | 0.002345 | 0.002594 | 0.999895 | 0.002345 | 0.903775 |
| total_reward | 0.065279 | 0.065323 | 1.000000 | 0.065279 | 25.162718 |
| original_env_reward | 0.010313 | 2.381671 | 1.000000 | 0.010313 | 3.975449 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_quality_reward | 32.510771 | 32.510771 | 0.153081 | 83.077111 | 1952 |
| attitude_penalty | -0.232635 | 0.232635 | -4.219355 | -0.006510 | 1952 |
| progress_reward | 1.203051 | 1.203212 | -0.157786 | 1.422032 | 1952 |
| total_reward | 33.481187 | 33.483011 | -0.484199 | 84.239887 | 1952 |

## Distribution
- score: mean=139.033146, min=104.909615, max=176.436820
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
