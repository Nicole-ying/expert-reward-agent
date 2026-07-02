# Training Feedback

## Training outcome
score=141.580265, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.004143 | 0.004459 | 0.996960 | 1.000000 |
| soft_landing_proxy | 0.214469 | 0.214469 | 0.428937 | 51.760557 |
| stability_penalty | -0.000493 | 0.000493 | 1.000000 | -0.118955 |
| total_reward | 0.218119 | 0.218495 | 1.000000 | 52.641603 |
| generated_reward | 0.218119 | 0.218495 | 1.000000 | 52.641603 |
| original_env_reward | -0.018222 | 1.581263 | 1.000000 | -4.397829 |

## Distribution
- score: mean=141.580265, min=116.915932, max=177.415322
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
