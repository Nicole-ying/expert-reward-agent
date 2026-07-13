# Training Feedback

## Training outcome
score=250.403608, len=1077.400000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| alive_bonus | 0.174791 | 0.174791 | 0.989832 | 0.189703 |
| progress_reward | 0.921391 | 0.926730 | 1.000000 | 1.000000 |
| stability_penalty | -0.036376 | 0.036376 | 1.000000 | -0.039480 |
| total_reward | 1.059806 | 1.069011 | 1.000000 | 1.150224 |
| generated_reward | 1.059806 | 1.069011 | 1.000000 | 1.150224 |
| original_env_reward | 0.131734 | 0.189971 | 1.000000 | 0.142973 |
| original_env_reward | 0.131734 | 0.189971 | 1.000000 | 0.142973 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=250.403608, min=-49.408814, max=281.502100
- episode_length: mean=1077.400000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
