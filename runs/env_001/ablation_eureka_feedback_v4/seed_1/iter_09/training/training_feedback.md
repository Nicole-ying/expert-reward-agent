# Training Feedback

## Final-policy outcome
score=-95.203726, len=826.650000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-204.395208, 191.838115]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 183.958386 | 97.8% | 97.8% | 3.5% |
| progress_reward | 1.181926 | 0.6% | 1.8% | 100.0% |
| orientation_penalty | -0.725174 | -0.4% | 0.4% | 100.0% |
| velocity_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
