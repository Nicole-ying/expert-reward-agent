# Training Feedback

## Final-policy outcome
score=151.317939, len=952.700000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[46.677893, 244.578232]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 61.585000 | 75.5% | 75.5% | 77.5% |
| progress | 13.732695 | 16.8% | 18.1% | 99.9% |
| angle_penalty | -2.757850 | -3.4% | 3.4% | 100.0% |
| speed_penalty | -2.048840 | -2.5% | 2.5% | 99.9% |
| angvel_penalty | -0.419048 | -0.5% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
