# Training Feedback

## Final-policy outcome
score=662.911633, len=621.950000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[35.255589, 1359.842825]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 3538.536875 | 91.2% | 91.4% | 100.0% |
| height_reward | 172.927110 | 4.5% | 4.5% | 100.0% |
| action_cost | -138.031585 | -3.6% | 3.6% | 100.0% |
| upright_penalty | -21.174185 | -0.5% | 0.5% | 7.3% |
| height_boundary_penalty | -0.283941 | -0.0% | 0.0% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
