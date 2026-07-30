# Training Feedback

## Final-policy outcome
score=-442.528422, len=837.250000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-1621.723953, -1.139540]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 736.098017 | 63.2% | 64.4% | 81.6% |
| upright_bonus | 261.941624 | 22.5% | 34.3% | 100.0% |
| lateral_penalty | -12.606909 | -1.1% | 1.1% | 76.4% |
| action_penalty | -2.556818 | -0.2% | 0.2% | 100.0% |
| height_penalty | -0.160128 | -0.0% | 0.0% | 19.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
