# Training Feedback

## Final-policy outcome
score=707.192269, len=223.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[699.126520, 715.139375]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 486.589535 | 99.1% | 99.1% | 100.0% |
| ang_vel_penalty | -3.187724 | -0.6% | 0.6% | 100.0% |
| posture_penalty | -1.372329 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
