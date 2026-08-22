# Training Feedback

## Final-policy outcome
score=-53.292398, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-90.412879, -9.533744]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| fuel_cost | -42.720000 | -63.7% | 63.7% | 85.4% |
| progress_reward | 13.278782 | 19.8% | 34.5% | 100.0% |
| attitude_penalty | -1.205911 | -1.8% | 1.8% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
