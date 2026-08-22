# Training Feedback

## Final-policy outcome
score=24.075442, len=128.450000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-135.627244, 140.847159]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_penalty | 121.201042 | 54.6% | 54.6% | 100.0% |
| upright_penalty | 52.068633 | 23.5% | 23.5% | 100.0% |
| velocity_reward | 28.131123 | 12.7% | 15.7% | 73.1% |
| angular_penalty | 11.080735 | 5.0% | 5.0% | 73.5% |
| lateral_penalty | 2.670008 | 1.2% | 1.2% | 73.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
