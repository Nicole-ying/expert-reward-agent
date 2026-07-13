# Training Feedback

## Training outcome
score=263.365361, len=1318.800000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| alive_bonus | 0.465762 | 0.465762 | 0.993467 | 0.641831 |
| progress_reward | 0.725677 | 0.729534 | 1.000000 | 1.000000 |
| stability_penalty | -0.021328 | 0.021328 | 1.000000 | -0.029391 |
| total_reward | 1.170111 | 1.176911 | 1.000000 | 1.612440 |
| generated_reward | 1.170111 | 1.176911 | 1.000000 | 1.612440 |
| original_env_reward | 0.112716 | 0.177540 | 1.000000 | 0.155325 |
| original_env_reward | 0.112716 | 0.177540 | 1.000000 | 0.155325 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=263.365361, min=260.325505, max=266.465812
- episode_length: mean=1318.800000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
