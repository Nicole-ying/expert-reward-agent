# Training Feedback

## Training outcome
score=120.136956, len=966.800000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| approach_quality_reward | 0.087956 | 0.087956 | 1.000000 | 0.087956 | 35.750430 |
| attitude_penalty | -0.000553 | 0.000553 | 0.999999 | -0.000553 | -0.224831 |
| contact_reward | 0.025029 | 0.025029 | 0.643187 | 0.038915 | 10.173416 |
| progress_reward | 0.002178 | 0.002460 | 0.999679 | 0.002178 | 0.885123 |
| total_reward | 0.114610 | 0.114637 | 1.000000 | 0.114610 | 46.584138 |
| original_env_reward | 0.038123 | 1.225731 | 1.000000 | 0.038123 | 15.495278 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_quality_reward | 48.223376 | 48.223376 | 0.207798 | 122.947646 | 1828 |
| attitude_penalty | -0.303163 | 0.303163 | -2.556174 | -0.003349 | 1828 |
| contact_reward | 13.727090 | 13.727090 | 0.000000 | 35.980000 | 1828 |
| progress_reward | 1.192707 | 1.203929 | -2.214701 | 1.421579 | 1828 |
| total_reward | 62.840010 | 62.841235 | -0.458426 | 157.219791 | 1828 |

## Distribution
- score: mean=120.136956, min=39.250020, max=157.799489
- episode_length: mean=966.800000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
