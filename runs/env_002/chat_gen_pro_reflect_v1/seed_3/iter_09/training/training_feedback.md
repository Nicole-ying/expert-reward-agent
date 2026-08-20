# Training Feedback

## Final-policy outcome
score=249.012097, len=842.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-31.017980, 307.079184]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 2769.261010 | 69.2% | 69.2% | 99.7% |
| balance_gate | 842.331004 | 21.0% | 21.0% | 100.0% |
| gait_quality | 275.650000 | 6.9% | 6.9% | 32.7% |
| gait_efficiency | 73.489056 | 1.8% | 1.8% | 32.5% |
| both_off_penalty | -22.980000 | -0.6% | 0.6% | 27.3% |
| action_penalty | -13.916344 | -0.3% | 0.3% | 100.0% |
| balance_modulation | -2.734179 | -0.1% | 0.1% | 100.0% |
| posture_penalty | -2.399942 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
