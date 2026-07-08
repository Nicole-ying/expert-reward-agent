# Training Feedback

## Training outcome
score=-107.359745, len=68.500000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| attitude_penalty | -0.000249 | 0.000249 | 1.000000 | -0.000249 | -0.014625 |
| landing_quality_reward | 0.004575 | 0.004575 | 0.008454 | 0.541194 | 0.269154 |
| progress_reward | 0.016064 | 0.016999 | 0.999991 | 0.016064 | 0.944972 |
| total_reward | 0.016730 | 0.017627 | 1.000000 | 0.016730 | 0.984178 |
| original_env_reward | -1.552170 | 2.413084 | 1.000000 | -1.552170 | -91.308725 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_penalty | -0.017459 | 0.017459 | -0.931127 | -0.001217 | 14289 |
| landing_quality_reward | 0.321330 | 0.321330 | 0.000000 | 0.956869 | 14289 |
| progress_reward | 1.127941 | 1.127963 | -0.157786 | 1.412644 | 14289 |
| total_reward | 1.174749 | 1.175083 | -0.655217 | 1.564718 | 14289 |

## Distribution
- score: mean=-107.359745, min=-123.927336, max=-78.933292
- episode_length: mean=68.500000
- early_terminal (<150 steps + score<-50): 20/20 (100%)
- errors: 0
