# Training Feedback

## Final-policy outcome
score=-75.713473, len=107.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-145.001253, 32.416633]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_guidance | 167.933840 | 95.3% | 97.7% | 99.2% |
| distance_progress | 2.425954 | 1.4% | 1.5% | 100.0% |
| tilt_penalty | -1.370367 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 11/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
