# Training Feedback

## Final-policy outcome
score=89.065223, len=805.900000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[35.028495, 212.743853]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settlement_bonus | 16726.003158 | 99.9% | 99.9% | 35.4% |
| speed_penalty_gated | -5.804138 | -0.0% | 0.0% | 100.0% |
| proximity_reward | 2.569321 | 0.0% | 0.0% | 100.0% |
| orientation_penalty | -1.970223 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
