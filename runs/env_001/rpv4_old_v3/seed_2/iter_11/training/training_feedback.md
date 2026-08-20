# Training Feedback

## Final-policy outcome
score=-7.587057, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-40.994947, 28.904234]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 26.426158 | 63.1% | 71.9% | 100.0% |
| fuel_penalty | -10.000000 | -23.9% | 23.9% | 100.0% |
| vel_damping | -1.478922 | -3.5% | 3.5% | 100.0% |
| angle_stab | -0.290908 | -0.7% | 0.7% | 100.0% |
| landing_quality | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
