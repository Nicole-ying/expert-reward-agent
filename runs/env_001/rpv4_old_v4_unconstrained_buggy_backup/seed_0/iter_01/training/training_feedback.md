# Training Feedback

## Final-policy outcome
score=-159.490304, len=303.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-274.805680, -104.968022]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 0.104125 | 21.1% | 93.2% | 100.0% |
| orientation_penalty | -0.033500 | -6.8% | 6.8% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| vel_damping | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
