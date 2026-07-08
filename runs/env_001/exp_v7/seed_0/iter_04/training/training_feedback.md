# Training Feedback

## Training outcome
score=-113.140622, len=68.400000, errors=0

## Component evidence

Column definitions:
- `mean`: per-step average of the component value (signed). Positive=reward, negative=penalty.
- `abs_mean`: per-step average of the absolute value. Measures magnitude regardless of sign.
- `nonzero_rate`: fraction of steps where the component value is non-zero (|value| > 1e-12).
- `abs_contrib_%`: this component's share of total reward magnitude. Computed as `100 * abs_mean_of_this_component / sum(abs_mean_of_all_components)`. All components sum to 100%. Use this to judge relative scale — a penalty with high abs_contrib_% may be dominating the learning signal.

| component | mean | abs_mean | nonzero_rate | abs_contrib_% |
|-----------|------|----------|-------------|--------------|
| distance_reward | -0.971496 | 0.971496 | 1.000000 | 17.527946 |
| soft_landing | 0.012206 | 0.012206 | 0.004317 | 0.220228 |
| stability_penalty | -0.025298 | 0.025298 | 1.000000 | 0.456430 |
| total_reward | -0.984588 | 1.007964 | 1.000000 | 18.185903 |
| generated_reward | -0.984588 | 1.007964 | 1.000000 | 18.185903 |
| original_env_reward | -1.738025 | 2.517629 | 1.000000 | 45.423590 |

## Distribution
- score: mean=-113.140622, min=-135.214739, max=-98.869787
- episode_length: mean=68.400000
- early_terminal (<150 steps + score<-50): 20/20 (100%)
- errors: 0
