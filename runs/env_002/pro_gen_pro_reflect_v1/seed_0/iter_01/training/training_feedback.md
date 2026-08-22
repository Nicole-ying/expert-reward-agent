# Training Feedback

## Final-policy outcome
score=313.860015, len=811.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[312.709792, 314.658906]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_speed_reward | 1007.176566 | 98.5% | 98.5% | 99.2% |
| action_cost | -13.487510 | -1.3% | 1.3% | 100.0% |
| upright_penalty | -2.225808 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
