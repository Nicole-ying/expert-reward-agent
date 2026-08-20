# Training Feedback

## Final-policy outcome
score=57.229343, len=920.650000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-2.451211, 250.390359]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 619.180103 | 99.1% | 99.1% | 48.4% |
| progress | 2.540391 | 0.4% | 0.5% | 100.0% |
| orientation_penalty | -2.068911 | -0.3% | 0.3% | 100.0% |
| velocity_penalty | -0.212533 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
