# Training Feedback

## Final-policy outcome
score=-5.086164, len=39.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-68.843575, 20.545362]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 60.197577 | 62.8% | 77.9% | 100.0% |
| lateral_penalty | -21.209757 | -22.1% | 22.1% | 100.0% |
| height_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| upright_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
