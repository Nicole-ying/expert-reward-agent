# Training Feedback

## Final-policy outcome
score=2.524524, len=13.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-28.387182, 17.529615]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 12.016889 | 59.3% | 59.3% | 70.5% |
| action_penalty | -4.989419 | -24.6% | 24.6% | 100.0% |
| lateral_penalty | -3.253387 | -16.1% | 16.1% | 100.0% |
| height_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
