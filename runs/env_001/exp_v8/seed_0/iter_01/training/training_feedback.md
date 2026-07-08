# Training Feedback

## Training outcome
score=-88.785487, len=69.000000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | 0.015989 | 0.016941 | 0.999990 | 1.000000 |
| soft_landing_proxy | 0.003192 | 0.003192 | 0.003192 | 0.199627 |
| stability_penalty | -0.058028 | 0.058028 | 1.000000 | -3.629324 |
| total_reward | -0.038848 | 0.045166 | 1.000000 | -2.429697 |
| generated_reward | -0.038848 | 0.045166 | 1.000000 | -2.429697 |
| original_env_reward | -1.456194 | 2.453977 | 1.000000 | -91.076577 |
| original_env_reward | -1.456194 | 2.453977 | 1.000000 | -91.076577 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=-88.785487, min=-119.653678, max=-63.373109
- episode_length: mean=69.000000
- early_terminal (<150 steps + score<-50): 20/20 (100%)
- errors: 0
