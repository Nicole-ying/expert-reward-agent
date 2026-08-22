# Training Feedback

## Final-policy outcome
score=-59.195713, len=394.200000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-109.433251, -4.257589]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| posture_gate | 268.462062 | 93.8% | 93.8% | 100.0% |
| progress_reward | 17.195273 | 6.0% | 6.1% | 98.6% |
| vertical_penalty | -0.134286 | -0.0% | 0.0% | 65.5% |
| air_penalty | -0.075000 | -0.0% | 0.0% | 1.3% |
| angular_penalty | -0.019512 | -0.0% | 0.0% | 65.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
