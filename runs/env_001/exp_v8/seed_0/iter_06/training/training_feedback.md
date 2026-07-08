# Training Feedback

## Training outcome
score=-251.555038, len=73.300000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | 0.139902 | 0.144890 | 1.000000 | 1.000000 |
| soft_landing_proxy | 0.000110 | 0.000110 | 0.000741 | 0.000783 |
| stability_penalty | -0.000000 | 0.000000 | 0.003508 | -0.000000 |
| total_reward | 0.140011 | 0.144879 | 1.000000 | 1.000783 |
| generated_reward | 0.140011 | 0.144879 | 1.000000 | 1.000783 |
| original_env_reward | -2.558133 | 4.061386 | 1.000000 | -18.285190 |
| original_env_reward | -2.558133 | 4.061386 | 1.000000 | -18.285190 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=-251.555038, min=-349.582181, max=-160.489005
- episode_length: mean=73.300000
- early_terminal (<150 steps + score<-50): 20/20 (100%)
- errors: 0
