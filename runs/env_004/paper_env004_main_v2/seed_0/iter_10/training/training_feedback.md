# Training Feedback

## Final-policy outcome
score=1346.906336, len=510.600000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[603.728417, 2591.786832]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_posture_reward | 786.995759 | 99.4% | 99.4% | 100.0% |
| ang_vel_penalty | 4.981088 | 0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
