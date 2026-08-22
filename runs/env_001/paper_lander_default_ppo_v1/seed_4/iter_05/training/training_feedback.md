# Training Feedback

## Final-policy outcome
score=197.603371, len=669.500000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[17.786682, 311.094322]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settlement_bonus | 715.781905 | 98.6% | 98.6% | 68.2% |
| speed_penalty_gated | -6.530935 | -0.9% | 0.9% | 99.3% |
| proximity_reward | 2.730791 | 0.4% | 0.4% | 99.1% |
| orientation_penalty | -0.773887 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
