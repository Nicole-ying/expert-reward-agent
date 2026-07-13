# Training Feedback

## Training outcome
score=130.668666, len=780.250000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| alive_bonus | 0.099079 | 0.099079 | 0.990792 | 0.059611 |
| progress_reward | 1.662086 | 1.662086 | 0.983013 | 1.000000 |
| stability_penalty | -0.044790 | 0.044790 | 1.000000 | -0.026948 |
| total_reward | 1.716375 | 1.717628 | 1.000000 | 1.032663 |
| generated_reward | 1.716375 | 1.717628 | 1.000000 | 1.032663 |
| original_env_reward | 0.120953 | 0.244040 | 1.000000 | 0.072772 |
| original_env_reward | 0.120953 | 0.244040 | 1.000000 | 0.072772 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=130.668666, min=-47.796038, max=272.721805
- episode_length: mean=780.250000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
