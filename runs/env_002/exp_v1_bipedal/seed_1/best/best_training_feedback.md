# Training Feedback

## Training outcome
score=250.895712, len=1357.900000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| alive_bonus | 0.497998 | 0.497998 | 0.995996 | 0.867781 |
| progress_reward | 0.573876 | 0.573876 | 0.976599 | 1.000000 |
| stability_penalty | -0.010121 | 0.010121 | 1.000000 | -0.017635 |
| total_reward | 1.061753 | 1.062286 | 1.000000 | 1.850145 |
| generated_reward | 1.061753 | 1.062286 | 1.000000 | 1.850145 |
| original_env_reward | 0.113551 | 0.191083 | 1.000000 | 0.197867 |
| original_env_reward | 0.113551 | 0.191083 | 1.000000 | 0.197867 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=250.895712, min=242.616030, max=257.639807
- episode_length: mean=1357.900000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
