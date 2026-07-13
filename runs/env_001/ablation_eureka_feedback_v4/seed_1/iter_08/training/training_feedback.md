# Training Feedback

## Final-policy outcome
score=-138.372823, len=668.300000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-213.278637, -47.881227]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_penalty | -34.067878 | -95.1% | 95.1% | 100.0% |
| progress_reward | 0.611853 | 1.7% | 4.2% | 100.0% |
| orientation_penalty | -0.248246 | -0.7% | 0.7% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
