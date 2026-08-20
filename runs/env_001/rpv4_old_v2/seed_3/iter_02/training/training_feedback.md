# Training Feedback

## Final-policy outcome
score=124.271173, len=604.200000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[-87.308558, 244.895182]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_reward | 852.000000 | 98.5% | 98.5% | 7.1% |
| orientation_stability_reward | -9.478931 | -1.1% | 1.1% | 100.0% |
| progress_reward | 1.877989 | 0.2% | 0.2% | 99.2% |
| velocity_damping_reward | -1.435409 | -0.2% | 0.2% | 99.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
