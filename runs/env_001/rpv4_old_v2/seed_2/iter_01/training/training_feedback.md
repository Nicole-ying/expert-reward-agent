# Training Feedback

## Final-policy outcome
score=-120.998945, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-150.975402, -99.021460]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -148.505275 | -51.5% | 51.5% | 100.0% |
| velocity_penalty | -93.225748 | -32.4% | 32.4% | 99.9% |
| angle_penalty | -43.986293 | -15.3% | 15.3% | 100.0% |
| success_proxy | 1.432261 | 0.5% | 0.5% | 0.1% |
| contact_quality | 0.766166 | 0.3% | 0.3% | 0.2% |
| thrust_penalty | -0.215000 | -0.1% | 0.1% | 6.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
