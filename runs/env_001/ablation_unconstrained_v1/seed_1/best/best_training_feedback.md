# Training Feedback

## Final-policy outcome
score=-108.688572, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.714740, -90.133068]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_penalty | -4.707838 | -53.1% | 53.1% | 100.0% |
| landing_bonus | 3.000000 | 33.8% | 33.8% | 0.4% |
| progress_reward | 1.119618 | 12.6% | 13.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
