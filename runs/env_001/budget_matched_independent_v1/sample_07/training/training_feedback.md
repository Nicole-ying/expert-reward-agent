# Training Feedback

## Final-policy outcome
score=96.178819, len=964.500000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[43.302274, 229.481994]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | 6.973724 | 71.9% | 73.7% | 99.8% |
| speed_penalty | -2.010120 | -20.7% | 20.7% | 99.8% |
| health_gate_modulation | -0.375674 | -3.9% | 3.9% | 91.0% |
| angle_hinge_penalty | -0.161980 | -1.7% | 1.7% | 1.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
