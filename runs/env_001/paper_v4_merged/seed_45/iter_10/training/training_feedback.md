# Training Feedback

## Final-policy outcome
score=-118.499137, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-153.146023, -80.846303]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| completion | 737.456844 | 99.3% | 99.3% | 100.0% |
| progress | 0.436870 | 0.1% | 0.3% | 100.0% |
| speed_penalty | -1.753082 | -0.2% | 0.2% | 2.5% |
| angle_penalty | -1.038953 | -0.1% | 0.1% | 2.3% |
| angvel_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
