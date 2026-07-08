# Training Feedback

## Training outcome
score=500.000000, len=500.000000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | 0.098606 | 0.098606 | 1.000000 | 1.000000 |
| stability_penalty | -0.000339 | 0.000339 | 1.000000 | -0.003437 |
| survival_bonus | 0.005000 | 0.005000 | 1.000000 | 0.050707 |
| total_reward | 0.103267 | 0.103267 | 1.000000 | 1.047270 |
| generated_reward | 0.103267 | 0.103267 | 1.000000 | 1.047270 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 10.141340 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 10.141340 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=500.000000, min=500.000000, max=500.000000
- episode_length: mean=500.000000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
