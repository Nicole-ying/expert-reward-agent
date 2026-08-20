# Training Feedback

## Final-policy outcome
score=-222.062615, len=148.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-263.950722, -163.671605]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| failure_penalty | -1.500000 | -35.4% | 35.4% | 0.2% |
| efficiency | -1.364000 | -32.2% | 32.2% | 45.9% |
| angle_penalty | -1.032716 | -24.4% | 24.4% | 34.8% |
| progress | 0.290442 | 6.9% | 8.0% | 97.2% |
| angvel_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 12/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
