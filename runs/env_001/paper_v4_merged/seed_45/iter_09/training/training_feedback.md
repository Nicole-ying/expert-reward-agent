# Training Feedback

## Final-policy outcome
score=-45.068935, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-76.752602, -8.359576]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 4.788842 | 42.2% | 57.1% | 100.0% |
| angle_penalty | -3.265621 | -28.7% | 28.7% | 3.8% |
| speed_penalty | -1.538650 | -13.5% | 13.5% | 2.0% |
| contact_reward | 0.050000 | 0.4% | 0.4% | 0.0% |
| angvel_penalty | -0.015091 | -0.1% | 0.1% | 0.2% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| completion | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
