# Training Feedback

## Final-policy outcome
score=78.110493, len=896.050000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[13.296212, 264.246128]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 681.500000 | 92.3% | 92.3% | 13.2% |
| stable_contact_bonus | 40.930000 | 5.5% | 5.5% | 45.7% |
| progress | 10.360270 | 1.4% | 1.7% | 100.0% |
| orientation_penalty | -1.709633 | -0.2% | 0.2% | 100.0% |
| velocity_penalty | -1.014798 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
