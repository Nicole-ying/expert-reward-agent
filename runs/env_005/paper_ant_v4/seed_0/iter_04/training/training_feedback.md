# Training Feedback

## Final-policy outcome
score=26.065180, len=181.850000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-421.722040, 184.219116]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_reward | 58.168597 | 36.5% | 56.3% | 56.6% |
| upright_penalty | 26.110944 | 16.4% | 16.4% | 100.0% |
| angular_penalty | 19.688462 | 12.3% | 12.3% | 56.7% |
| height_penalty | 15.339334 | 9.6% | 9.6% | 100.0% |
| lateral_penalty | 8.654746 | 5.4% | 5.4% | 56.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
