# Training Feedback

## External evaluation
- score: 158.822193
- episode_length: 728.400000 (mean)
- range: [93.826443, 208.689203]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| landing_shaping | 1.624923 | 1.624923 | 0.542595 | 0.000000 | 4.990018 |
| progress_reward | 0.203996 | 0.219075 | 0.998808 | -1.641093 | 2.296980 |
| stability_penalty | -0.037143 | 0.037143 | 1.000000 | -0.456914 | -0.000000 |
| total_reward | 1.791776 | 1.809284 | 1.000000 | -1.843835 | 4.990014 |
| generated_reward | 1.791776 | 1.809284 | 1.000000 | -1.843835 | 4.990014 |
| original_env_reward | -0.172914 | 1.689840 | 1.000000 | -100.000000 | 136.593785 |

## Signals
partial_progress; penalty_dominance:landing_shaping; penalty_dominance:generated_reward; penalty_dominance:original_env_reward
