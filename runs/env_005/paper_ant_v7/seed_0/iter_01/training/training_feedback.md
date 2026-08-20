# Training Feedback

## Final-policy outcome
score=-12.040768, len=80.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-89.466250, 30.378798]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 90.458902 | 84.9% | 95.4% | 100.0% |
| action_cost | -3.656153 | -3.4% | 3.4% | 100.0% |
| upright_penalty | -0.861365 | -0.8% | 0.8% | 100.0% |
| height_penalty | -0.377965 | -0.4% | 0.4% | 4.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
