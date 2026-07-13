# Training Feedback

## Final-policy outcome
score=-124.805825, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335003, -105.977095]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 1.113752 | 49.0% | 50.8% | 100.0% |
| vertical_penalty | -0.927947 | -40.9% | 40.9% | 74.5% |
| contact_bonus | 0.162615 | 7.2% | 7.2% | 3.4% |
| fuel_cost | -0.025000 | -1.1% | 1.1% | 1.8% |
| angle_penalty | -0.002331 | -0.1% | 0.1% | 3.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
