# Training Feedback

## Final-policy outcome
score=464.772331, len=907.700000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-1108.309802, 709.741333]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1633.219925 | 62.1% | 62.2% | 99.5% |
| _height_gate | 862.375083 | 32.8% | 32.8% | 99.8% |
| lateral_penalty | -89.757130 | -3.4% | 3.4% | 95.3% |
| upright_penalty | -39.163092 | -1.5% | 1.5% | 5.5% |
| action_penalty | -2.595718 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
