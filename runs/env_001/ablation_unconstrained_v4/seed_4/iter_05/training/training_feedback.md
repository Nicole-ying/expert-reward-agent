# Training Feedback

## Final-policy outcome
score=-125.821079, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335003, -106.352030]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_penalty | 69.486721 | 65.4% | 65.4% | 83.0% |
| progress | 27.801015 | 26.1% | 27.1% | 100.0% |
| contact_reward | 6.158329 | 5.8% | 5.8% | 1.2% |
| ang_penalty | 1.025256 | 1.0% | 1.0% | 18.3% |
| fuel_penalty | 0.850000 | 0.8% | 0.8% | 2.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
