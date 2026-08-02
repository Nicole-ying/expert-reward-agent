# Training Feedback

## Final-policy outcome
score=-52.725987, len=323.500000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-94.679558, 0.556516]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 175.944875 | 79.3% | 79.4% | 100.0% |
| air_penalty | -34.680991 | -15.6% | 15.6% | 71.2% |
| posture_penalty | -6.256749 | -2.8% | 2.8% | 4.3% |
| action_cost | -4.756964 | -2.1% | 2.1% | 100.0% |
| ang_vel_penalty | -0.017643 | -0.0% | 0.0% | 78.1% |
| vertical_speed_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
