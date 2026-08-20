# Training Feedback

## Final-policy outcome
score=293.915478, len=863.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[20.538459, 319.041092]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 2983.354637 | 69.7% | 69.7% | 98.8% |
| balance_gate | 859.971774 | 20.1% | 20.1% | 100.0% |
| gait_quality | 303.700000 | 7.1% | 7.1% | 35.2% |
| gait_efficiency | 83.617934 | 2.0% | 2.0% | 34.6% |
| both_off_penalty | -23.935000 | -0.6% | 0.6% | 27.7% |
| balance_modulation | -12.310073 | -0.3% | 0.3% | 99.9% |
| action_penalty | -11.218423 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
