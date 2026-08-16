# Training Feedback

## Final-policy outcome
score=-117.237675, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-140.279795, -98.837163]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_damping | -21.082070 | -44.1% | 44.1% | 100.0% |
| proximity_reward | -19.972254 | -41.8% | 41.8% | 100.0% |
| landing_bonus | 2.500000 | 5.2% | 5.2% | 0.4% |
| action_reward | 1.744500 | 3.7% | 4.3% | 100.0% |
| contact_reward | 1.625000 | 3.4% | 3.4% | 3.1% |
| orientation_penalty | -0.540341 | -1.1% | 1.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
