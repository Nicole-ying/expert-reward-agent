# Training Feedback

## Training outcome
score=240.905342, len=415.150000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| center_bonus | 0.114038 | 0.114038 | 0.740887 | 2.297248 |
| progress_reward | 0.049641 | 0.054063 | 0.998966 | 1.000000 |
| soft_landing_proxy | 0.372611 | 0.372611 | 0.519768 | 7.506118 |
| total_reward | 0.536290 | 0.538518 | 0.999998 | 10.803365 |
| generated_reward | 0.536290 | 0.538518 | 0.999998 | 10.803365 |
| original_env_reward | -0.108050 | 1.652333 | 1.000000 | -2.176627 |
| original_env_reward | -0.108050 | 1.652333 | 1.000000 | -2.176627 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=240.905342, min=202.336528, max=274.576486
- episode_length: mean=415.150000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
