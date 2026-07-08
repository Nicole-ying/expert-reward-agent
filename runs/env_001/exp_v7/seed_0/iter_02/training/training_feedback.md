# Training Feedback

## Training outcome
score=-115.294647, len=68.400000, errors=0

## Component evidence

Column definitions:
- `mean`: per-step average of the component value (signed). Positive=reward, negative=penalty.
- `abs_mean`: per-step average of the absolute value. Measures magnitude regardless of sign.
- `nonzero_rate`: fraction of steps where the component value is non-zero (|value| > 1e-12).
- `abs_contrib_%`: this component's share of total reward magnitude. Computed as `100 * abs_mean_of_this_component / sum(abs_mean_of_all_components)`. All components sum to 100%. Use this to judge relative scale — a penalty with high abs_contrib_% may be dominating the learning signal.

| component | mean | abs_mean | nonzero_rate | abs_contrib_% |
|-----------|------|----------|-------------|--------------|
| distance_reward | -0.969099 | 0.969099 | 1.000000 | 16.837481 |
| soft_landing_continuous | 0.012902 | 0.012902 | 0.005527 | 0.224167 |
| stability_penalty | -0.026946 | 0.026946 | 1.000000 | 0.468169 |
| total_reward | -0.983142 | 1.007392 | 1.000000 | 17.502804 |
| generated_reward | -0.983142 | 1.007392 | 1.000000 | 17.502804 |
| original_env_reward | -1.928712 | 2.731872 | 1.000000 | 47.464575 |

## Distribution
- score: mean=-115.294647, min=-141.582367, max=-95.059093
- episode_length: mean=68.400000
- early_terminal (<150 steps + score<-50): 20/20 (100%)
- errors: 0
