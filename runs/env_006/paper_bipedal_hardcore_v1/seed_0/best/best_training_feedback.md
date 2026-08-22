# Training Feedback

## Final-policy outcome
score=-42.416168, len=406.500000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-85.774709, 30.355815]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward_gated | 136.530978 | 75.3% | 75.3% | 84.7% |
| contact_transition_reward | 40.190000 | 22.2% | 22.2% | 99.0% |
| action_cost | -4.415159 | -2.4% | 2.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
