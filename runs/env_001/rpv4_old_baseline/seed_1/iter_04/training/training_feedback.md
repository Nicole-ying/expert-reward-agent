# Training Feedback

## Final-policy outcome
score=-113.098330, len=994.800000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-202.561246, -64.811817]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| pos_attract | 828.522430 | 71.7% | 71.7% | 100.0% |
| vel_penalty | -177.560764 | -15.4% | 15.4% | 100.0% |
| step_cost | -99.480000 | -8.6% | 8.6% | 100.0% |
| thrust_cost | -41.787500 | -3.6% | 3.6% | 84.0% |
| stability_penalty | -6.495798 | -0.6% | 0.6% | 100.0% |
| progress_reward | 2.010667 | 0.2% | 0.2% | 66.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
