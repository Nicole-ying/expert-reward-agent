# Training Feedback

## Final-policy outcome
score=-87.194134, len=143.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-225.198024, 27.283910]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_speed_gate | 127.298497 | 96.4% | 96.4% | 100.0% |
| progress_shaping | 1.081386 | 0.8% | 1.2% | 100.0% |
| shaped_progress | 0.872964 | 0.7% | 1.0% | 100.0% |
| landing_contact_reward | 1.260690 | 1.0% | 1.0% | 11.9% |
| action_cost | -0.659000 | -0.5% | 0.5% | 45.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 9/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
