# Training Feedback

## Training outcome
score=-4345.464756, len=383.100000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| landing_reward | 0.000120 | 0.000120 | 0.000012 | 10.000000 | 0.000831 |
| progress_reward | 0.143181 | 0.143849 | 1.000000 | 0.143181 | 0.995353 |
| total_reward | 0.143301 | 0.143969 | 1.000000 | 0.143301 | 0.996184 |
| original_env_reward | -5.786956 | 6.159514 | 1.000000 | -5.786956 | -40.229274 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_reward | 0.083102 | 0.083102 | 0.000000 | 80.000000 | 1444 |
| progress_reward | 99.159312 | 99.159312 | 0.744308 | 330.170053 | 1444 |
| total_reward | 99.242415 | 99.242415 | 0.744308 | 330.170053 | 1444 |

## Distribution
- score: mean=-4345.464756, min=-10799.475542, max=-395.312781
- episode_length: mean=383.100000
- early_terminal (<150 steps + score<-50): 6/20 (30%)
- errors: 0
