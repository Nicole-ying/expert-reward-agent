# Training Feedback

## Final-policy outcome
score=216.495195, len=1132.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-19.724232, 286.639660]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 738.219311 | 99.3% | 99.3% | 98.3% |
| upright_stability_penalty | -4.970219 | -0.7% | 0.7% | 2.8% |
| vertical_oscillation_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
