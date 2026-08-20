# Training Feedback

## Final-policy outcome
score=180.206789, len=553.300000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-27.690641, 277.689663]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 1811.573083 | 93.7% | 93.7% | 100.0% |
| speed_tracking_reward | -120.757898 | -6.2% | 6.2% | 100.0% |
| progress_reward | 1.243109 | 0.1% | 0.1% | 99.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
