# Training Feedback

## Final-policy outcome
score=-45.740324, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-76.464295, -4.664443]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 16.733679 | 28.0% | 42.0% | 100.0% |
| fuel_penalty | -19.617000 | -32.8% | 32.8% | 65.4% |
| proximity_warmth | 10.573632 | 17.7% | 17.7% | 100.0% |
| velocity_penalty | -3.238598 | -5.4% | 5.4% | 100.0% |
| angle_penalty | -1.277877 | -2.1% | 2.1% | 100.0% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_readiness | 0.000000 | 0.0% | 0.0% | 0.0% |
| perfect_landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
