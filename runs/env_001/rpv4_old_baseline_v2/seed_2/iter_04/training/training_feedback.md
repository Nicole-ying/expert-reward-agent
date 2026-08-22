# Training Feedback

## Final-policy outcome
score=-122.771736, len=504.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-206.198272, 209.407677]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 405.369141 | 75.8% | 75.8% | 1.9% |
| vel_penalty | -77.893223 | -14.6% | 14.6% | 99.9% |
| descent_reward | 24.496232 | 4.6% | 4.6% | 97.1% |
| engine_cost | -16.397500 | -3.1% | 3.1% | 78.4% |
| approach_reward | 2.527262 | 0.5% | 1.1% | 99.8% |
| attitude_penalty | -4.312890 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
