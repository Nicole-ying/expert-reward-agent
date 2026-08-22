# Training Feedback

## Final-policy outcome
score=-626.753640, len=88.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-746.907896, -417.525435]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -73.046921 | -82.0% | 82.0% | 100.0% |
| position_reward | -12.230638 | -13.7% | 13.7% | 100.0% |
| height_cost | -3.035729 | -3.4% | 3.4% | 100.0% |
| orientation_cost | -0.635061 | -0.7% | 0.7% | 100.0% |
| engine_penalty | -0.150600 | -0.2% | 0.2% | 84.7% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| vy_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
