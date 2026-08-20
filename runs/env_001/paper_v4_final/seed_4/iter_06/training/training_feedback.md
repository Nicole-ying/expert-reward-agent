# Training Feedback

## Final-policy outcome
score=-60.085154, len=71.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-88.715384, -29.733855]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 22.166389 | 73.9% | 77.0% | 100.0% |
| survival | -5.700000 | -19.0% | 19.0% | 100.0% |
| landing_prox | 0.614710 | 2.1% | 2.1% | 0.7% |
| stability | -0.401287 | -1.3% | 1.3% | 100.0% |
| fuel | -0.184500 | -0.6% | 0.6% | 25.9% |
| failure_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 15/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
