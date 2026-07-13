# Training Feedback

## Final-policy outcome
score=113.983536, len=786.950000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[11.522281, 251.414530]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_event | 691.545869 | 84.8% | 84.8% | 8.9% |
| soft_landing | 116.557006 | 14.3% | 14.3% | 82.2% |
| energy_penalty | -6.076500 | -0.7% | 0.7% | 77.2% |
| proximity | 1.333524 | 0.2% | 0.2% | 71.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
