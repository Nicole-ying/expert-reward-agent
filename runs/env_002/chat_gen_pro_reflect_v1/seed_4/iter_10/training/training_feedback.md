# Training Feedback

## Final-policy outcome
score=173.310030, len=738.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-4.955567, 314.605430]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 994.439767 | 51.7% | 51.7% | 100.0% |
| gated_forward | 521.218304 | 27.1% | 27.1% | 99.3% |
| forward_reward | 387.486075 | 20.2% | 20.2% | 99.3% |
| energy_penalty | -19.649036 | -1.0% | 1.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
