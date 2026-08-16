# Training Feedback

## Final-policy outcome
score=-120.427683, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-146.678292, -98.057243]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 13.396807 | 76.4% | 79.1% | 100.0% |
| attitude_penalty | -1.696977 | -9.7% | 9.7% | 100.0% |
| settled_reward | 0.750000 | 4.3% | 4.3% | 0.4% |
| height_penalty | -0.635954 | -3.6% | 3.6% | 100.0% |
| contact_discovery | 0.510000 | 2.9% | 2.9% | 1.9% |
| fuel_penalty | -0.076500 | -0.4% | 0.4% | 3.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
