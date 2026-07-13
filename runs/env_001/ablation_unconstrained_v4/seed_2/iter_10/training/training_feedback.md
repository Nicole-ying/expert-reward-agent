# Training Feedback

## Final-policy outcome
score=-296.340200, len=494.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-366.488749, -138.894322]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_reward | 848.513376 | 78.0% | 78.0% | 100.0% |
| alignment | 185.195502 | 17.0% | 17.0% | 100.0% |
| lat_penalty | -35.967463 | -3.3% | 3.3% | 41.8% |
| att_penalty | -13.333919 | -1.2% | 1.2% | 100.0% |
| angvel_penalty | -4.107528 | -0.4% | 0.4% | 100.0% |
| soft_landing_penalty | -0.459776 | -0.0% | 0.0% | 2.3% |
| progress_y | 0.017674 | 0.0% | 0.0% | 2.3% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
