# Training Feedback

## Final-policy outcome
score=-3.172451, len=12.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-10.799333, 5.256371]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vertical_penalty | -36.319887 | -52.2% | 52.2% | 100.0% |
| angular_penalty | -15.130354 | -21.7% | 21.7% | 100.0% |
| forward_reward | 3.786578 | 5.4% | 15.3% | 100.0% |
| lateral_penalty | -7.521742 | -10.8% | 10.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
