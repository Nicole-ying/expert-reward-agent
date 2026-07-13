# Training Feedback

## Final-policy outcome
score=-107.520418, len=91.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-137.610522, -64.680270]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_bonus | 4.114216 | 68.3% | 68.3% | 0.8% |
| distance_progress | 1.199019 | 19.9% | 24.2% | 100.0% |
| posture_penalty | -0.451073 | -7.5% | 7.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 17/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
