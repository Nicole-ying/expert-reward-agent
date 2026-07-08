# Training Feedback

## Training outcome
score=186.810738, len=608.450000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| landing_quality_bonus | 0.187986 | 0.187986 | 0.357100 | 0.526425 | 0.660178 |
| progress_reward | 0.270995 | 0.284751 | 0.997381 | 0.271706 | 0.951690 |
| total_reward | 0.458981 | 0.470881 | 0.999994 | 0.458984 | 1.611868 |
| original_env_reward | -0.820535 | 2.580426 | 1.000000 | -0.820535 | -2.881589 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_quality_bonus | 27.472421 | 27.472421 | 0.000000 | 795.350101 | 6855 |
| progress_reward | 39.651121 | 39.651121 | 8.398154 | 42.610680 | 6855 |
| total_reward | 67.123542 | 67.123542 | 8.398154 | 837.561602 | 6855 |

## Distribution
- score: mean=186.810738, min=-21.675135, max=278.056784
- episode_length: mean=608.450000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
