# Training Feedback

## Final-policy outcome
score=-24.964255, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-50.610096, 13.490928]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 686.511964 | 99.5% | 99.5% | 100.0% |
| progress_reward | 2.267035 | 0.3% | 0.4% | 100.0% |
| attitude_penalty | -1.082570 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
