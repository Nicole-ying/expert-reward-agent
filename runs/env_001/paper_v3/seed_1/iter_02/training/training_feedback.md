# Training Feedback

## Final-policy outcome
score=26.552706, len=349.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-169.709550, 267.202764]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 176.449042 | 52.4% | 52.4% | 13.5% |
| goal_proximity | -147.357246 | -43.8% | 43.8% | 100.0% |
| velocity_penalty | -10.688626 | -3.2% | 3.2% | 99.5% |
| orientation_penalty | -2.315943 | -0.7% | 0.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
