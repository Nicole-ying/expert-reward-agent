# Training Feedback

## Final-policy outcome
score=202.646384, len=748.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-38.866660, 312.734803]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 2408.242656 | 69.5% | 69.5% | 99.7% |
| balance_gate | 739.583823 | 21.3% | 21.3% | 100.0% |
| gait_quality | 208.000000 | 6.0% | 6.0% | 27.8% |
| gait_efficiency | 52.860244 | 1.5% | 1.5% | 27.5% |
| balance_modulation | -33.356289 | -1.0% | 1.0% | 99.9% |
| both_off_penalty | -14.885000 | -0.4% | 0.4% | 19.9% |
| action_penalty | -10.563939 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
