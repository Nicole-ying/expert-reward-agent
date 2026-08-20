# Training Feedback

## Final-policy outcome
score=225.882125, len=479.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[199.061371, 261.871767]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 346.757918 | 73.2% | 73.2% | 100.0% |
| contact_reward | 70.211244 | 14.8% | 14.8% | 7.8% |
| landing_speed_penalty | -56.494578 | -11.9% | 11.9% | 99.8% |
| settlement_penalty | -0.333686 | -0.1% | 0.1% | 7.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
