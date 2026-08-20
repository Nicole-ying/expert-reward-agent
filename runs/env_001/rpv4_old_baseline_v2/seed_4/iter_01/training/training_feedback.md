# Training Feedback

## Final-policy outcome
score=139.525725, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[110.154355, 173.022953]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_bonus | 250.151797 | 90.4% | 90.4% | 73.7% |
| fuel_penalty | -18.853000 | -6.8% | 6.8% | 94.3% |
| stability_penalty | -4.708082 | -1.7% | 1.7% | 100.0% |
| progress_reward | 2.707649 | 1.0% | 1.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
