# Training Feedback

## Final-policy outcome
score=292.628537, len=987.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[162.235469, 305.908973]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 1512.652111 | 53.9% | 53.9% | 100.0% |
| gated_forward | 766.033119 | 27.3% | 27.3% | 99.6% |
| forward_reward | 500.132293 | 17.8% | 17.8% | 99.6% |
| energy_penalty | -30.085844 | -1.1% | 1.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
