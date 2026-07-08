# Training Feedback

## Training outcome
score=466.150000, len=466.150000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | -0.001244 | 0.003207 | 1.000000 | -1.000000 |
| stability_penalty | -0.003930 | 0.003930 | 1.000000 | -3.159049 |
| total_reward | -0.005174 | 0.006307 | 1.000000 | -4.159049 |
| generated_reward | -0.005174 | 0.006307 | 1.000000 | -4.159049 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 803.846372 |
| original_env_reward | 1.000000 | 1.000000 | 1.000000 | 803.846372 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=466.150000, min=156.000000, max=500.000000
- episode_length: mean=466.150000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
