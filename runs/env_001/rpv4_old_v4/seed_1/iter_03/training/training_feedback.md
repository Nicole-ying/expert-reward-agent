# Training Feedback

## Final-policy outcome
score=-19.584728, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-52.158335, 16.601510]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 484.379909 | 94.1% | 94.1% | 83.0% |
| proximity_progress | 24.077010 | 4.7% | 5.5% | 100.0% |
| attitude_penalty | -1.203096 | -0.2% | 0.2% | 100.0% |
| landing_speed | -1.085062 | -0.2% | 0.2% | 83.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
