# Training Feedback

## Training outcome
score=144.886927, len=888.050000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| alive_bonus | 0.500000 | 0.500000 | 1.000000 | 0.907851 |
| progress_reward | 0.550751 | 0.555075 | 1.000000 | 1.000000 |
| stability_penalty | -0.037162 | 0.037162 | 1.000000 | -0.067476 |
| total_reward | 1.013589 | 1.015578 | 1.000000 | 1.840375 |
| generated_reward | 1.013589 | 1.015578 | 1.000000 | 1.840375 |
| original_env_reward | 0.126340 | 0.180160 | 1.000000 | 0.229396 |
| original_env_reward | 0.126340 | 0.180160 | 1.000000 | 0.229396 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=144.886927, min=-102.721316, max=281.310537
- episode_length: mean=888.050000
- early_terminal (<150 steps + score<-50): 2/20 (10%)
- errors: 0
