# Training Feedback

## Training outcome
score=153.606278, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| dist_gate | 0.530557 | 0.530557 | 1.000000 | 27.800503 |
| landing_proxy | 0.462081 | 0.462081 | 0.585062 | 24.212416 |
| progress_reward | 0.019084 | 0.020500 | 0.998793 | 1.000000 |
| stability_penalty | -0.001271 | 0.001271 | 1.000000 | -0.066597 |
| total_reward | 0.479894 | 0.481039 | 1.000000 | 25.145819 |
| generated_reward | 0.479894 | 0.481039 | 1.000000 | 25.145819 |
| original_env_reward | -0.122405 | 1.593658 | 1.000000 | -6.413860 |

## Distribution
- score: mean=153.606278, min=120.542448, max=183.580521
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
