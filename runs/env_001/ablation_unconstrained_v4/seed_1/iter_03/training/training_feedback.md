# Training Feedback

## Final-policy outcome
score=-62.672049, len=979.900000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-91.889328, -15.928720]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| step_penalty | -48.995000 | -61.3% | 61.3% | 100.0% |
| contact_reward | 20.250000 | 25.4% | 25.4% | 0.3% |
| distance_progress | 4.138454 | 5.2% | 9.1% | 100.0% |
| speed_penalty | -3.276417 | -4.1% | 4.1% | 100.0% |
| angle_penalty | -0.103067 | -0.1% | 0.1% | 100.0% |
| angvel_penalty | -0.008897 | -0.0% | 0.0% | 100.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
