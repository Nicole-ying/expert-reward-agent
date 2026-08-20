# Training Feedback

## Training outcome
score=137.606613, len=1000.000000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | 0.034361 | 0.036802 | 0.999804 | 1.000000 |
| soft_landing_proxy | 0.210295 | 0.210295 | 0.481814 | 6.120186 |
| stability_penalty | -0.000000 | 0.000000 | 0.003508 | -0.000000 |
| total_reward | 0.244655 | 0.246697 | 0.999993 | 7.120186 |
| generated_reward | 0.244655 | 0.246697 | 0.999993 | 7.120186 |
| original_env_reward | -0.247575 | 1.769447 | 1.000000 | -7.205159 |
| original_env_reward | -0.247575 | 1.769447 | 1.000000 | -7.205159 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=137.606613, min=107.956019, max=171.250968
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
