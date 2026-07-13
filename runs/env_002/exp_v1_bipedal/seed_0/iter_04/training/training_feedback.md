# Training Feedback

## Training outcome
score=269.685355, len=1279.350000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| alive_bonus | 0.455284 | 0.455284 | 0.994842 | 0.385751 |
| progress_reward | 1.180255 | 1.185287 | 1.000000 | 1.000000 |
| stability_penalty | -0.025204 | 0.025204 | 1.000000 | -0.021355 |
| total_reward | 1.610335 | 1.615708 | 1.000000 | 1.364396 |
| generated_reward | 1.610335 | 1.615708 | 1.000000 | 1.364396 |
| original_env_reward | 0.129632 | 0.178786 | 1.000000 | 0.109834 |
| original_env_reward | 0.129632 | 0.178786 | 1.000000 | 0.109834 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=269.685355, min=265.060738, max=272.648874
- episode_length: mean=1279.350000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
