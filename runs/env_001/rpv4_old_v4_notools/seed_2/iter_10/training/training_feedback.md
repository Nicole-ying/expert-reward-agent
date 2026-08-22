# Training Feedback

## Final-policy outcome
score=261.882375, len=298.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[234.865795, 296.811010]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_holding_bonus | 42.550000 | 63.0% | 63.0% | 14.3% |
| progress | 13.659842 | 20.2% | 20.7% | 95.0% |
| step_penalty | -8.947500 | -13.3% | 13.3% | 100.0% |
| velocity_penalty | -1.024798 | -1.5% | 1.5% | 94.9% |
| orientation_penalty | -0.999905 | -1.5% | 1.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
