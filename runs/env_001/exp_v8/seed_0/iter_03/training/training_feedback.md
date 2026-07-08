# Training Feedback

## Training outcome
score=169.757416, len=922.350000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | 0.042595 | 0.045449 | 0.997967 | 1.000000 |
| soft_landing_proxy | 0.293871 | 0.293871 | 0.415430 | 6.899227 |
| stability_penalty | -0.000000 | 0.000000 | 0.003508 | -0.000000 |
| total_reward | 0.336465 | 0.338809 | 0.999904 | 7.899227 |
| generated_reward | 0.336465 | 0.338809 | 0.999904 | 7.899227 |
| original_env_reward | -0.315241 | 1.970220 | 1.000000 | -7.400940 |
| original_env_reward | -0.315241 | 1.970220 | 1.000000 | -7.400940 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=169.757416, min=104.131736, max=293.316014
- episode_length: mean=922.350000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
