# Training Feedback

## Final-policy outcome
score=245.371132, len=264.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-4.750641, 310.695734]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 55.998883 | 79.9% | 79.9% | 18.0% |
| progress | 13.003884 | 18.6% | 19.4% | 95.0% |
| angle_penalty | -0.399370 | -0.6% | 0.6% | 100.0% |
| angvel_penalty | -0.106031 | -0.2% | 0.2% | 93.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
