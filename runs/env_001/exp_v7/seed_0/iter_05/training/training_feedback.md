# Training Feedback

## Training outcome
score=216.838994, len=527.150000, errors=0

## Component evidence

Column definitions:
- `mean`: per-step average of the component value (signed). Positive=reward, negative=penalty.
- `abs_mean`: per-step average of the absolute value. Measures magnitude regardless of sign.
- `nonzero_rate`: fraction of steps where the component value is non-zero (|value| > 1e-12).
- `abs_contrib_%`: this component's share of total reward magnitude. Computed as `100 * abs_mean_of_this_component / sum(abs_mean_of_all_components)`. All components sum to 100%. Use this to judge relative scale — a penalty with high abs_contrib_% may be dominating the learning signal.

| component | mean | abs_mean | nonzero_rate | abs_contrib_% |
|-----------|------|----------|-------------|--------------|
| progress_reward | 0.002365 | 0.002553 | 0.996653 | 0.090013 |
| soft_landing_reward | 0.335935 | 0.335935 | 0.335935 | 11.846313 |
| stability_penalty | -0.001071 | 0.001071 | 1.000000 | 0.037765 |
| total_reward | 0.337229 | 0.337673 | 1.000000 | 11.907631 |
| generated_reward | 0.337229 | 0.337673 | 1.000000 | 11.907631 |
| original_env_reward | -0.146123 | 1.820868 | 1.000000 | 64.210645 |

## Distribution
- score: mean=216.838994, min=75.633064, max=281.041340
- episode_length: mean=527.150000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0
