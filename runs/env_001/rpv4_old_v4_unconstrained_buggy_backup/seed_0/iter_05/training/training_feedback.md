# Training Feedback

## Final-policy outcome
score=-121.303998, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-145.573911, -99.218497]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 13.391426 | 56.9% | 58.9% | 100.0% |
| vel_damping | -7.113626 | -30.2% | 30.2% | 100.0% |
| landing_bonus | 1.273579 | 5.4% | 5.4% | 2.2% |
| orientation_penalty | -1.143416 | -4.9% | 4.9% | 100.0% |
| action_penalty | -0.160000 | -0.7% | 0.7% | 4.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
