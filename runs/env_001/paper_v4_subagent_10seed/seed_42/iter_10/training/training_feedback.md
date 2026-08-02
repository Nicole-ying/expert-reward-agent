# Training Feedback

## Final-policy outcome
score=6.146694, len=960.600000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-56.338766, 153.316800]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 801.151878 | 99.3% | 99.3% | 100.0% |
| soft_landing | 4.672819 | 0.6% | 0.6% | 1.6% |
| progress_reward | 0.630752 | 0.1% | 0.1% | 70.0% |
| attitude_penalty | -0.033199 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
