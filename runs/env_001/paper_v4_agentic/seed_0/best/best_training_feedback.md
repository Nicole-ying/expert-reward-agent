# Training Feedback

## Final-policy outcome
score=232.573910, len=460.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[200.292128, 264.009958]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_bonus | 139.370649 | 98.8% | 98.8% | 100.0% |
| progress_reward | 1.380049 | 1.0% | 1.0% | 96.7% |
| landing_safety_penalty | 0.235785 | 0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
