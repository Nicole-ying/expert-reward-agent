# Training Feedback

## External evaluation
- score: -230.170851
- episode_length: 77.400000 (mean)
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_reward | -0.091284 | 0.091284 | 1.000000 | -0.172670 | -0.000020 |
| progress_reward | 0.904659 | 0.937327 | 0.999997 | -1.754445 | 3.030356 |
| stability_penalty | -0.053275 | 0.053275 | 0.230069 | -1.053463 | 0.000000 |
| total_reward | 0.760101 | 0.827861 | 1.000000 | -2.066579 | 2.533566 |
| generated_reward | 0.760101 | 0.827861 | 1.000000 | -2.066579 | 2.533566 |
| original_env_reward | -2.974001 | 4.452664 | 1.000000 | -100.000000 | 132.283320 |

## Signals
early_failure_or_crash; penalty_dominance:generated_reward; penalty_dominance:original_env_reward
