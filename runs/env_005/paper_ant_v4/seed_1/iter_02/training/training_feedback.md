# Training Feedback

## Final-policy outcome
score=-224.845626, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-391.757258, 585.070983]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 675.563807 | 22.2% | 68.1% | 95.5% |
| upright_stability | 970.297156 | 31.8% | 31.8% | 100.0% |
| lateral_drift_penalty | -1.207796 | -0.0% | 0.0% | 83.5% |
| height_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
