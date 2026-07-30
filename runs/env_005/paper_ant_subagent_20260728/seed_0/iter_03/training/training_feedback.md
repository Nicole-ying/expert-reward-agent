# Training Feedback

## Final-policy outcome
score=1839.706014, len=981.500000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[896.580849, 2166.796600]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 3421.016521 | 90.3% | 90.3% | 98.2% |
| upright_bonus | 186.142573 | 4.9% | 5.1% | 100.0% |
| lateral_penalty | -170.575763 | -4.5% | 4.5% | 98.7% |
| action_penalty | -3.307078 | -0.1% | 0.1% | 100.0% |
| height_penalty | -0.030798 | -0.0% | 0.0% | 2.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
