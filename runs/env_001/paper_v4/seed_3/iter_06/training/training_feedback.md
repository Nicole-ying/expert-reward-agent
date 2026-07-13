# Training Feedback

## Final-policy outcome
score=247.206661, len=410.400000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[150.236728, 303.323417]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 85.633935 | 83.7% | 83.7% | 50.2% |
| progress | 6.777173 | 6.6% | 6.8% | 97.5% |
| velocity_penalty | -5.564843 | -5.4% | 5.4% | 97.9% |
| orientation_penalty | -2.714379 | -2.7% | 2.7% | 100.0% |
| engine_penalty | -1.396250 | -1.4% | 1.4% | 68.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
