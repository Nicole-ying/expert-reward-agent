# Training Feedback

## Final-policy outcome
score=-10.108862, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-43.149389, 24.819673]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_penalty | -7.837401 | -47.6% | 47.6% | 100.0% |
| upright_penalty | -5.113069 | -31.0% | 31.0% | 100.0% |
| goal_proximity | 2.922025 | 17.7% | 21.4% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
