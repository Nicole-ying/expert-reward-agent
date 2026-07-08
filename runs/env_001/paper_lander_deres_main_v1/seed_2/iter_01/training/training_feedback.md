# Training Feedback

## Training outcome
score=-161.078864, len=77.850000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| landing_bonus | 0.254424 | 0.254424 | 0.025442 | 10.000000 | 0.269106 |
| orientation_penalty | -0.020067 | 0.020067 | 1.000000 | -0.020067 | -0.021225 |
| progress_reward | -0.945442 | 0.945442 | 1.000000 | -0.945442 | -1.000000 |
| total_reward | -0.711085 | 1.210921 | 1.000000 | -0.711085 | -0.752119 |
| original_env_reward | -2.215704 | 4.003521 | 1.000000 | -2.215704 | -2.343564 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_bonus | 19.001265 | 19.001265 | 0.000000 | 7370.000000 | 13437 |
| orientation_penalty | -1.498547 | 1.498547 | -12.503569 | -0.047418 | 13437 |
| progress_reward | -70.595356 | 70.595356 | -521.598896 | -41.901344 | 13437 |
| total_reward | -53.092638 | 69.965459 | -530.999269 | 7177.928743 | 13437 |

## Distribution
- score: mean=-161.078864, min=-299.212645, max=-5.792269
- episode_length: mean=77.850000
- early_terminal (<150 steps + score<-50): 16/20 (80%)
- errors: 0
