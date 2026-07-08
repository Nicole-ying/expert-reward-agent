# Training Feedback

## Training outcome
score=125.203289, len=888.150000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| attitude_penalty | -0.000857 | 0.000857 | 1.000000 | -0.000857 | -0.298774 |
| descent_incentive | 0.016962 | 0.016962 | 0.611992 | 0.027716 | 5.916209 |
| landing_settle | 0.088859 | 0.088859 | 0.626063 | 0.141933 | 30.993179 |
| progress_reward | 0.002615 | 0.002867 | 0.999544 | 0.002616 | 0.912132 |
| total_reward | 0.107580 | 0.107778 | 1.000000 | 0.107580 | 37.522747 |
| original_env_reward | 0.006421 | 1.233703 | 1.000000 | 0.006421 | 2.239461 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_penalty | -0.410419 | 0.410419 | -3.451744 | -0.002579 | 2093 |
| descent_incentive | 8.114789 | 8.114789 | 0.454825 | 15.142271 | 2093 |
| landing_settle | 42.570026 | 42.570026 | 0.000000 | 126.761316 | 2093 |
| progress_reward | 1.251348 | 1.251499 | -0.157786 | 1.420788 | 2093 |
| total_reward | 51.525744 | 51.525936 | -0.200391 | 135.855236 | 2093 |

## Distribution
- score: mean=125.203289, min=-110.815629, max=267.092641
- episode_length: mean=888.150000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
