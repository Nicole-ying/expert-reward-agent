# Training Feedback

## Final-policy outcome
score=-180.102474, len=479.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-219.225934, -108.767180]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_damping | -8.963955 | -44.4% | 44.4% | 20.7% |
| action_reward | -8.576000 | -42.5% | 42.5% | 64.1% |
| orientation_penalty | -1.530771 | -7.6% | 7.6% | 100.0% |
| progress_reward | -0.485253 | -2.4% | 5.6% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
