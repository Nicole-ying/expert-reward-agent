# Training Feedback

## Final-policy outcome
score=172.387758, len=734.950000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[58.511867, 235.798164]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_completion | 762.242505 | 93.5% | 93.5% | 37.8% |
| soft_landing_speed_penalty | -38.992614 | -4.8% | 4.8% | 99.9% |
| approach_progress | 6.144100 | 0.8% | 0.9% | 99.3% |
| orientation_penalty | -6.747322 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
