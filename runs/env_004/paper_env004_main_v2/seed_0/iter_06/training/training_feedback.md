# Training Feedback

## Final-policy outcome
score=142.497023, len=71.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[139.062013, 145.061886]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| acceleration_reward | 6.500836 | 84.6% | 94.3% | 100.0% |
| vertical_penalty | -0.366090 | -4.8% | 4.8% | 100.0% |
| ang_vel_penalty | -0.040153 | -0.5% | 0.5% | 99.9% |
| posture_penalty | -0.030559 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
