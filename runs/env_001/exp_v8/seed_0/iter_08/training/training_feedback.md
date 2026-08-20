# Training Feedback

## Training outcome
score=245.281525, len=374.900000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| center_bonus | 0.114894 | 0.114894 | 0.732319 | 4.152320 |
| progress_reward | 0.027670 | 0.029791 | 0.999319 | 1.000000 |
| soft_landing_proxy | 0.391318 | 0.391318 | 0.520510 | 14.142416 |
| total_reward | 0.533881 | 0.534920 | 0.999979 | 19.294735 |
| generated_reward | 0.533881 | 0.534920 | 0.999979 | 19.294735 |
| original_env_reward | -0.076945 | 1.606194 | 1.000000 | -2.780842 |
| original_env_reward | -0.076945 | 1.606194 | 1.000000 | -2.780842 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=245.281525, min=105.171554, max=294.904996
- episode_length: mean=374.900000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
