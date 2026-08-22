# Training Feedback

## Final-policy outcome
score=197.111424, len=834.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-79.425800, 302.082889]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 393.607308 | 96.1% | 96.1% | 100.0% |
| action_cost | -15.945011 | -3.9% | 3.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
