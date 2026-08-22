# Training Feedback

## Final-policy outcome
score=-73.707864, len=731.500000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-660.017042, 186.445660]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 2258.693178 | 72.2% | 74.7% | 91.8% |
| upright_penalty | -760.389820 | -24.3% | 24.3% | 99.7% |
| action_cost | -31.571790 | -1.0% | 1.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
