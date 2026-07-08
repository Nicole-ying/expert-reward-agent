# Training Feedback

## Training outcome
score=-153.053812, len=133.900000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| landing_bonus | 1.212373 | 1.212373 | 0.253987 | 4.773368 | 1.548481 |
| orientation_penalty | -0.018956 | 0.018956 | 1.000000 | -0.018956 | -0.024211 |
| progress_reward | -0.782943 | 0.782943 | 1.000000 | -0.782943 | -1.000000 |
| total_reward | 0.410474 | 1.909171 | 1.000000 | 0.410474 | 0.524270 |
| original_env_reward | -1.591747 | 3.602428 | 1.000000 | -1.591747 | -2.033030 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_bonus | 119.283934 | 119.283934 | 0.000000 | 8001.381261 | 10190 |
| orientation_penalty | -1.865463 | 1.865463 | -18.949881 | -0.048128 | 10190 |
| progress_reward | -77.057154 | 77.057154 | -574.683072 | -41.965808 | 10190 |
| total_reward | 40.361317 | 161.360132 | -460.289887 | 7905.675253 | 10190 |

## Distribution
- score: mean=-153.053812, min=-249.691188, max=-32.767124
- episode_length: mean=133.900000
- early_terminal (<150 steps + score<-50): 17/20 (85%)
- errors: 0
