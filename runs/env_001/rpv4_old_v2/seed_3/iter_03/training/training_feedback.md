# Training Feedback

## Final-policy outcome
score=110.436884, len=528.550000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-51.736246, 219.059345]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy_reward | 359.974666 | 91.1% | 91.1% | 24.5% |
| action_penalty | -15.682500 | -4.0% | 4.0% | 98.9% |
| orientation_stability_reward | -15.550877 | -3.9% | 3.9% | 100.0% |
| progress_reward | 3.643578 | 0.9% | 1.0% | 99.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
