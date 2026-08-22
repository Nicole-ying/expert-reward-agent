# Training Feedback

## Final-policy outcome
score=44.342176, len=915.150000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-221.901624, 95.612410]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_progress | 487.369996 | 95.3% | 95.3% | 48.9% |
| distance_delta | 13.517811 | 2.6% | 3.0% | 100.0% |
| engine_penalty | -9.151500 | -1.8% | 1.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
