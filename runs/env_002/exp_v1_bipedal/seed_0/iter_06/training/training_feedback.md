# Training Feedback

## Training outcome
score=251.914036, len=1195.850000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| gait_bonus | 0.346973 | 0.346973 | 0.996762 | 0.576391 |
| progress_reward | 0.601975 | 0.605476 | 1.000000 | 1.000000 |
| stability_penalty | -0.040546 | 0.040546 | 1.000000 | -0.067355 |
| total_reward | 0.908402 | 0.910606 | 1.000000 | 1.509036 |
| generated_reward | 0.908402 | 0.910606 | 1.000000 | 1.509036 |
| original_env_reward | 0.121821 | 0.197875 | 1.000000 | 0.202368 |
| original_env_reward | 0.121821 | 0.197875 | 1.000000 | 0.202368 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=251.914036, min=76.665034, max=264.748147
- episode_length: mean=1195.850000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
