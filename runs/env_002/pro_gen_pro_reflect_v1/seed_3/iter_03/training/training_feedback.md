# Training Feedback

## Final-policy outcome
score=311.392247, len=793.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[45.162215, 326.631250]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 484.553173 | 97.3% | 97.4% | 100.0% |
| action_cost | -10.958240 | -2.2% | 2.2% | 100.0% |
| tilt_penalty | -2.075793 | -0.4% | 0.4% | 100.0% |
| angular_velocity_penalty | -0.158565 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
