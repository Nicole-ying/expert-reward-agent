# Training Feedback

## Final-policy outcome
score=135.219760, len=285.100000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-43.255408, 287.305222]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_bonus | 161.366598 | 76.2% | 76.2% | 23.5% |
| proximity_progress | 44.360581 | 21.0% | 22.5% | 97.4% |
| attitude_penalty | -2.601216 | -1.2% | 1.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
