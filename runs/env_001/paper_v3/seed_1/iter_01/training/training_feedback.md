# Training Feedback

## Final-policy outcome
score=109.342157, len=675.500000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-42.899449, 298.326804]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -117.655932 | -68.9% | 68.9% | 100.0% |
| contact_reward | 43.761620 | 25.6% | 25.6% | 3.5% |
| velocity_penalty | -8.405496 | -4.9% | 4.9% | 99.4% |
| orientation_penalty | -0.833308 | -0.5% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
