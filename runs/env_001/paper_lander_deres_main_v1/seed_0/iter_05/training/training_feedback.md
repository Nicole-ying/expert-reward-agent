# Training Feedback

## Training outcome
score=139.740956, len=1000.000000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| approach_quality_reward | 0.088804 | 0.088804 | 1.000000 | 0.088804 | 36.567786 |
| attitude_penalty | -0.000505 | 0.000505 | 1.000000 | -0.000505 | -0.207907 |
| landing_proxy | 0.154965 | 0.154965 | 0.622587 | 0.248904 | 63.811306 |
| progress_reward | 0.002182 | 0.002428 | 0.999706 | 0.002182 | 0.898397 |
| total_reward | 0.245446 | 0.245481 | 1.000000 | 0.245446 | 101.069582 |
| original_env_reward | 0.046635 | 1.210071 | 1.000000 | 0.046635 | 19.203357 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_quality_reward | 48.555690 | 48.555690 | 0.209403 | 127.102068 | 1832 |
| attitude_penalty | -0.276033 | 0.276033 | -2.848695 | -0.008264 | 1832 |
| landing_proxy | 84.744687 | 84.744687 | 0.000000 | 241.742410 | 1832 |
| progress_reward | 1.192060 | 1.199265 | -2.940459 | 1.420731 | 1832 |
| total_reward | 134.216405 | 134.217345 | -0.458426 | 370.054551 | 1832 |

## Distribution
- score: mean=139.740956, min=106.098209, max=176.239484
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
