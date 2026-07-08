# Training Feedback

## Training outcome
score=146.361474, len=1000.000000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| approach_quality_reward | 0.093574 | 0.093574 | 1.000000 | 0.093574 | 40.188992 |
| attitude_penalty | -0.000503 | 0.000503 | 1.000000 | -0.000503 | -0.216165 |
| progress_reward | 0.002089 | 0.002328 | 0.999854 | 0.002089 | 0.897058 |
| total_reward | 0.095160 | 0.095192 | 1.000000 | 0.095160 | 40.869885 |
| original_env_reward | 0.031884 | 2.404357 | 1.000000 | 0.031884 | 13.693968 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_quality_reward | 54.432782 | 54.432782 | 0.229621 | 125.237945 | 1722 |
| attitude_penalty | -0.292920 | 0.292920 | -3.161573 | -0.008264 | 1722 |
| progress_reward | 1.214756 | 1.217027 | -0.871415 | 1.421205 | 1722 |
| total_reward | 55.354618 | 55.355772 | -0.458426 | 126.172365 | 1722 |

## Distribution
- score: mean=146.361474, min=115.341605, max=178.943227
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
