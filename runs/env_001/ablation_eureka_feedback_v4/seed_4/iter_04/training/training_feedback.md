# Training Feedback

## Final-policy outcome
score=179.801154, len=795.650000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[103.192945, 259.694899]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 61.550977 | 83.6% | 83.6% | 100.0% |
| velocity_penalty | -6.512647 | -8.8% | 8.8% | 13.2% |
| approach_progress | 2.793646 | 3.8% | 3.9% | 99.1% |
| angle_stability | -2.730752 | -3.7% | 3.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
