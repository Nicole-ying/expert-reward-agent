# Training Feedback

## Final-policy outcome
score=-108.894981, len=68.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-145.893547, -60.648700]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_progress | 33.503052 | 48.4% | 50.2% | 100.0% |
| soft_landing_bonus | 30.000000 | 43.4% | 43.4% | 0.2% |
| attitude_penalty | -2.608744 | -3.8% | 3.8% | 100.0% |
| speed_penalty | -1.845046 | -2.7% | 2.7% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
