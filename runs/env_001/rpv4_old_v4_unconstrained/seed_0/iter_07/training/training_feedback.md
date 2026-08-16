# Training Feedback

## Final-policy outcome
score=83.186927, len=766.550000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-46.813458, 300.924232]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 261.744270 | 81.0% | 81.0% | 5.1% |
| contact_reward | 36.575000 | 11.3% | 11.3% | 5.1% |
| action_reward | -19.433000 | -6.0% | 6.0% | 72.2% |
| progress_reward | 2.350517 | 0.7% | 0.9% | 99.4% |
| vel_damping | -2.206279 | -0.7% | 0.7% | 13.4% |
| orientation_penalty | -0.406323 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
