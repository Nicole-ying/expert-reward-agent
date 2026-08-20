# Training Feedback

## Final-policy outcome
score=130.571559, len=778.900000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[49.468530, 285.456556]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_landing_bonus | 1116.612188 | 97.9% | 97.9% | 9.6% |
| progress | 13.330918 | 1.2% | 1.3% | 99.8% |
| landing_reward | 5.669467 | 0.5% | 0.5% | 97.9% |
| angle_penalty | -4.227434 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
