# Training Feedback

## Training outcome
score=-108.773725, len=68.450000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | 0.016043 | 0.016982 | 0.999998 | 1.000000 |
| soft_landing_proxy | 0.003080 | 0.003080 | 0.003080 | 0.191990 |
| stability_penalty | -0.006062 | 0.006062 | 1.000000 | -0.377833 |
| total_reward | 0.013062 | 0.015507 | 1.000000 | 0.814157 |
| generated_reward | 0.013062 | 0.015507 | 1.000000 | 0.814157 |
| original_env_reward | -1.575244 | 2.424714 | 1.000000 | -98.186655 |
| original_env_reward | -1.575244 | 2.424714 | 1.000000 | -98.186655 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=-108.773725, min=-124.224441, max=-91.779191
- episode_length: mean=68.450000
- early_terminal (<150 steps + score<-50): 20/20 (100%)
- errors: 0
