# Training Feedback

## Final-policy outcome
score=-155.975043, len=154.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-180.244147, -114.227696]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 1.582203 | 21.1% | 59.8% | 100.0% |
| action_cost | -1.804000 | -24.1% | 24.1% | 58.4% |
| angle_penalty | -1.140523 | -15.2% | 15.2% | 100.0% |
| landing_velocity_penalty | -0.066830 | -0.9% | 0.9% | 5.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 10/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
