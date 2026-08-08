# Training Feedback

## Final-policy outcome
score=-62.759712, len=359.700000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-97.533098, -22.101961]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward_gated | 90.336951 | 79.4% | 79.4% | 80.3% |
| contact_transition_reward | 17.059082 | 15.0% | 15.6% | 78.9% |
| action_cost | -5.713999 | -5.0% | 5.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 5/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
