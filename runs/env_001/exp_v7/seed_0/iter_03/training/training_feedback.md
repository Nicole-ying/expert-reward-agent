# Training Feedback

## Training outcome
score=-1368.988936, len=177.600000, errors=0

## Component evidence

Column definitions:
- `mean`: per-step average of the component value (signed). Positive=reward, negative=penalty.
- `abs_mean`: per-step average of the absolute value. Measures magnitude regardless of sign.
- `nonzero_rate`: fraction of steps where the component value is non-zero (|value| > 1e-12).
- `abs_contrib_%`: this component's share of total reward magnitude. Computed as `100 * abs_mean_of_this_component / sum(abs_mean_of_all_components)`. All components sum to 100%. Use this to judge relative scale — a penalty with high abs_contrib_% may be dominating the learning signal.

| component | mean | abs_mean | nonzero_rate | abs_contrib_% |
|-----------|------|----------|-------------|--------------|
| contact_bonus | 0.000750 | 0.000750 | 0.004858 | 0.011621 |
| potential_shaping | 0.126059 | 0.126599 | 1.000000 | 1.962533 |
| total_reward | 0.126809 | 0.127322 | 1.000000 | 1.973750 |
| generated_reward | 0.126809 | 0.127322 | 1.000000 | 1.973750 |
| original_env_reward | -5.606683 | 6.068778 | 1.000000 | 94.078345 |

## Distribution
- score: mean=-1368.988936, min=-8273.598986, max=-356.722928
- episode_length: mean=177.600000
- early_terminal (<150 steps + score<-50): 15/20 (75%)
- errors: 0
