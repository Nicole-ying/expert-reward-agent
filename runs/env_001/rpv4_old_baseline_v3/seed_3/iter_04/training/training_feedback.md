# Training Feedback

## Final-policy outcome
score=-363.304825, len=130.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-445.746360, -261.895134]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | -15.936317 | -62.7% | 69.4% | 100.0% |
| orient_cost | -3.501467 | -13.8% | 13.8% | 100.0% |
| speed_cost | -3.231261 | -12.7% | 12.7% | 27.4% |
| fuel_cost | -0.808000 | -3.2% | 3.2% | 61.9% |
| descent_reward | 0.231960 | 0.9% | 0.9% | 6.4% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 15/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
