# Training Feedback

## Final-policy outcome
score=-114.093212, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-139.982445, -94.359884]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 3.354443 | 43.0% | 44.6% | 100.0% |
| speed_penalty | -1.264842 | -16.2% | 16.2% | 100.0% |
| attitude_penalty | -1.016799 | -13.0% | 13.0% | 100.0% |
| descent_incentive | 0.860983 | 11.0% | 11.0% | 94.3% |
| landing_reward | 0.536060 | 6.9% | 6.9% | 3.1% |
| altitude_bonus | 0.440928 | 5.7% | 5.7% | 100.0% |
| fuel_penalty | -0.200000 | -2.6% | 2.6% | 5.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
