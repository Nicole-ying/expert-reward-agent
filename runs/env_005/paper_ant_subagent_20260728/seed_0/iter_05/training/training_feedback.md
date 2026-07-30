# Training Feedback

## Final-policy outcome
score=-73.653375, len=700.300000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-1236.070538, 450.304330]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 957.916732 | 81.4% | 82.2% | 80.2% |
| upright_bonus | 99.451698 | 8.5% | 11.5% | 100.0% |
| ang_penalty | -47.747347 | -4.1% | 4.1% | 48.1% |
| lateral_penalty | -23.421660 | -2.0% | 2.0% | 45.1% |
| action_penalty | -2.195791 | -0.2% | 0.2% | 100.0% |
| height_penalty | -0.107194 | -0.0% | 0.0% | 15.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
