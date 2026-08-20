# Training Feedback

## Final-policy outcome
score=213.705914, len=913.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-58.217024, 300.779514]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward_reward | 400.811413 | 96.6% | 96.6% | 99.3% |
| energy_penalty | -14.212596 | -3.4% | 3.4% | 100.0% |
| angle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
