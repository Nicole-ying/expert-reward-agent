# Training Feedback

## Final-policy outcome
score=-254.396704, len=304.600000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-1071.312110, -3.619734]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_factor | 241.103428 | 48.1% | 48.1% | 100.0% |
| upright_factor | 159.390888 | 31.8% | 31.8% | 53.2% |
| forward_speed_reward | 87.635861 | 17.5% | 17.5% | 53.2% |
| survival_reward | 13.608738 | 2.7% | 2.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
