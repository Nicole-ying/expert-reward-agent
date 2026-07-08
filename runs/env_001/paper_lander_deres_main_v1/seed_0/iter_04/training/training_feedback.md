# Training Feedback

## Training outcome
score=128.653997, len=910.250000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| approach_quality_reward | 0.093119 | 0.093119 | 1.000000 | 0.093119 | 36.774364 |
| attitude_penalty | -0.000491 | 0.000491 | 1.000000 | -0.000491 | -0.194023 |
| contact_bonus | 0.055852 | 0.055852 | 0.654338 | 0.085357 | 22.057043 |
| progress_reward | 0.002294 | 0.002532 | 0.999759 | 0.002295 | 0.906100 |
| total_reward | 0.150774 | 0.150795 | 1.000000 | 0.150774 | 59.543484 |
| original_env_reward | 0.042052 | 1.518914 | 1.000000 | 0.042052 | 16.607069 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_quality_reward | 48.599795 | 48.599795 | 0.197759 | 126.994708 | 1920 |
| attitude_penalty | -0.256316 | 0.256316 | -2.740604 | -0.006799 | 1920 |
| contact_bonus | 29.150246 | 29.150246 | 0.000000 | 81.091039 | 1920 |
| progress_reward | 1.197038 | 1.197203 | -0.157786 | 1.421083 | 1920 |
| total_reward | 78.690763 | 78.692043 | -0.458426 | 208.952285 | 1920 |

## Distribution
- score: mean=128.653997, min=-45.193185, max=176.649476
- episode_length: mean=910.250000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
