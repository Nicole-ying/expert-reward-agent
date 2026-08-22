# Training Feedback

## Final-policy outcome
score=-43.005721, len=757.950000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-95.227887, 65.688807]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_transition_reward | 187.740000 | 49.9% | 49.9% | 99.1% |
| forward_reward_gated | 177.779274 | 47.3% | 47.3% | 73.9% |
| action_cost | -10.488204 | -2.8% | 2.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
