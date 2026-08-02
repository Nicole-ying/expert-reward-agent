# Training Feedback

## Final-policy outcome
score=12.246215, len=127.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-9.926946, 35.112643]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 130.000000 | 76.8% | 76.8% | 0.5% |
| progress | 25.654848 | 15.2% | 15.6% | 100.0% |
| survival | -10.168000 | -6.0% | 6.0% | 100.0% |
| stability | -1.657673 | -1.0% | 1.0% | 100.0% |
| fuel | -0.999500 | -0.6% | 0.6% | 78.6% |
| failure_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
