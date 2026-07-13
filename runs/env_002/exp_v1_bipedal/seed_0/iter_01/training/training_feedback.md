# Training Feedback

## Training outcome
score=244.945917, len=1385.750000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| alive_bonus | 0.497027 | 0.497027 | 0.994054 | 0.869421 |
| progress_reward | 0.571676 | 0.575681 | 1.000000 | 1.000000 |
| stability_penalty | -0.015351 | 0.015351 | 1.000000 | -0.026853 |
| total_reward | 1.053352 | 1.056374 | 1.000000 | 1.842568 |
| generated_reward | 1.053352 | 1.056374 | 1.000000 | 1.842568 |
| original_env_reward | 0.117302 | 0.175417 | 1.000000 | 0.205189 |
| original_env_reward | 0.117302 | 0.175417 | 1.000000 | 0.205189 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=244.945917, min=242.129347, max=247.501914
- episode_length: mean=1385.750000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
