# Training Feedback

## Final-policy outcome
score=-76.561074, len=975.150000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-109.556126, -13.998983]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_penalty | -12.561269 | -71.7% | 71.7% | 100.0% |
| upright_penalty | -3.714416 | -21.2% | 21.2% | 100.0% |
| goal_proximity | 0.635053 | 3.6% | 7.2% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
