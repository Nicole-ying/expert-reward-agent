# Training Feedback

## Final-policy outcome
score=-54.867317, len=392.300000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-617.403098, 261.337561]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 244.146425 | 35.2% | 45.4% | 100.0% |
| upright_penalty | -215.078098 | -31.0% | 31.0% | 34.1% |
| height_reward | 94.751990 | 13.7% | 13.7% | 100.0% |
| action_cost | -69.175846 | -10.0% | 10.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
