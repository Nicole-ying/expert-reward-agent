# Training Feedback

## Training outcome
score=-111.870902, len=68.450000, errors=0

## Component evidence

Column definitions:
- `mean`: per-step average of the component value (signed). Positive=reward, negative=penalty.
- `abs_mean`: per-step average of the absolute value. Measures magnitude regardless of sign.
- `nonzero_rate`: fraction of steps where the component value is non-zero (|value| > 1e-12).
- `abs_contrib_%`: this component's share of total reward magnitude. Computed as `100 * abs_mean_of_this_component / sum(abs_mean_of_all_components)`. All components sum to 100%. Use this to judge relative scale — a penalty with high abs_contrib_% may be dominating the learning signal.

| component | mean | abs_mean | nonzero_rate | abs_contrib_% |
|-----------|------|----------|-------------|--------------|
| distance_reward | -0.970523 | 0.970523 | 1.000000 | 17.112529 |
| soft_landing_proxy | 0.028500 | 0.028500 | 0.002850 | 0.502514 |
| stability_penalty | -0.025649 | 0.025649 | 1.000000 | 0.452248 |
| total_reward | -0.967673 | 1.024070 | 1.000000 | 18.056669 |
| generated_reward | -0.967673 | 1.024070 | 1.000000 | 18.056669 |
| original_env_reward | -1.750103 | 2.598609 | 1.000000 | 45.819371 |

## Distribution
- score: mean=-111.870902, min=-125.263486, max=-95.638526
- episode_length: mean=68.450000
- early_terminal (<150 steps + score<-50): 20/20 (100%)
- errors: 0
