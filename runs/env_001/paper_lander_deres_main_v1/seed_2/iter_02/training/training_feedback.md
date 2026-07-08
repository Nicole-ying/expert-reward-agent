# Training Feedback

## Training outcome
score=-120.870252, len=77.850000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| landing_bonus | 0.182515 | 0.182515 | 0.043834 | 4.163815 | 0.195111 |
| orientation_penalty | -0.019643 | 0.019643 | 1.000000 | -0.019643 | -0.020998 |
| progress_reward | -0.935447 | 0.935447 | 1.000000 | -0.935447 | -1.000000 |
| total_reward | -0.772574 | 1.116855 | 1.000000 | -0.772574 | -0.825888 |
| original_env_reward | -2.012927 | 3.975818 | 1.000000 | -2.012927 | -2.151836 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_bonus | 13.862996 | 13.862996 | 0.000000 | 5135.527088 | 13212 |
| orientation_penalty | -1.491944 | 1.491944 | -10.450242 | -0.045714 | 13212 |
| progress_reward | -71.040877 | 71.040877 | -732.480769 | -42.055543 | 13212 |
| total_reward | -58.669826 | 67.053610 | -740.443053 | 5010.905581 | 13212 |

## Distribution
- score: mean=-120.870252, min=-255.029558, max=7.874621
- episode_length: mean=77.850000
- early_terminal (<150 steps + score<-50): 14/20 (70%)
- errors: 0
