# Training Feedback

## Final-policy outcome
score=-277.904767, len=986.650000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-367.212175, -196.249790]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 1030.828718 | 65.7% | 65.9% | 96.8% |
| upright_bonus | 459.124095 | 29.3% | 31.1% | 100.0% |
| lateral_penalty | -44.013497 | -2.8% | 2.8% | 97.9% |
| action_penalty | -2.882555 | -0.2% | 0.2% | 100.0% |
| height_penalty | -0.071885 | -0.0% | 0.0% | 5.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
