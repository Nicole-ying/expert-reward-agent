# Training Feedback

## Final-policy outcome
score=36.301928, len=368.550000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-160.196659, 203.906568]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 794.867101 | 92.3% | 92.3% | 44.7% |
| dist_penalty | -35.795525 | -4.2% | 4.2% | 100.0% |
| vel_penalty | -14.307612 | -1.7% | 1.7% | 100.0% |
| ang_penalty | -9.820004 | -1.1% | 1.1% | 100.0% |
| fuel_penalty | -6.165000 | -0.7% | 0.7% | 83.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
