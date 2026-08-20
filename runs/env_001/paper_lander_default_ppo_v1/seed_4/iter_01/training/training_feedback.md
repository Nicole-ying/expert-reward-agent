# Training Feedback

## Final-policy outcome
score=-65.966209, len=72.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-130.629645, 9.737831]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | -68.654744 | -86.0% | 86.0% | 100.0% |
| speed_penalty_gated | -10.860412 | -13.6% | 13.6% | 100.0% |
| contact_bonus | 0.199273 | 0.2% | 0.2% | 1.0% |
| orientation_penalty | -0.118358 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 13/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
