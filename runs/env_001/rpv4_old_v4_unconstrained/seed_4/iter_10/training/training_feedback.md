# Training Feedback

## Final-policy outcome
score=-120.213481, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-145.758655, -101.571810]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_delta | 3.758681 | 63.1% | 65.6% | 100.0% |
| stability_penalty | -1.393097 | -23.4% | 23.4% | 100.0% |
| contact_bonus | 0.607491 | 10.2% | 10.2% | 1.3% |
| action_penalty | -0.049500 | -0.8% | 0.8% | 7.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
