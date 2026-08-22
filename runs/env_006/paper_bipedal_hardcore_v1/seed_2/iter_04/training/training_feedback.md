# Training Feedback

## Final-policy outcome
score=-74.846412, len=376.950000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-139.905142, -37.793537]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terrain_gate | 190.975104 | 63.1% | 63.1% | 100.0% |
| terrain_roughness | 79.704034 | 26.3% | 26.3% | 100.0% |
| forward_reward | 30.297565 | 10.0% | 10.1% | 97.8% |
| balance_penalty | -1.357288 | -0.4% | 0.4% | 4.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 8/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
