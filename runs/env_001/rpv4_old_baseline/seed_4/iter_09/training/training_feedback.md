# Training Feedback

## Final-policy outcome
score=-90.354168, len=885.350000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[-156.811541, -36.039427]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| engine_penalty | -118.907500 | -47.9% | 47.9% | 95.9% |
| survival_bonus | 44.267500 | 17.8% | 17.8% | 100.0% |
| speed_match_penalty | -33.228459 | -13.4% | 13.4% | 100.0% |
| speed_penalty | -32.773227 | -13.2% | 13.2% | 100.0% |
| centering_penalty | -15.145661 | -6.1% | 6.1% | 100.0% |
| approach_reward | 1.250873 | 0.5% | 0.9% | 100.0% |
| angle_penalty | -1.001996 | -0.4% | 0.4% | 100.0% |
| angvel_penalty | -0.460781 | -0.2% | 0.2% | 100.0% |
| landing_success_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
