# Training Feedback

## Final-policy outcome
score=-393.521243, len=66.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-440.859872, -324.395242]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_penalty | -6.671216 | -41.9% | 41.9% | 100.0% |
| progress_reward | 3.383667 | 21.2% | 21.9% | 100.0% |
| landing_reward | 2.500000 | 15.7% | 15.7% | 0.1% |
| action_penalty | -2.110000 | -13.2% | 13.2% | 63.6% |
| time_penalty | -0.664000 | -4.2% | 4.2% | 100.0% |
| crash_penalty | -0.500000 | -3.1% | 3.1% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
