# Training Feedback

## Final-policy outcome
score=5.701264, len=11.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-9.930565, 13.462387]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 10.456869 | 60.5% | 60.5% | 71.6% |
| action_penalty | -3.630627 | -21.0% | 21.0% | 100.0% |
| lateral_penalty | -2.146437 | -12.4% | 12.4% | 100.0% |
| descend_penalty | -1.040935 | -6.0% | 6.0% | 41.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
