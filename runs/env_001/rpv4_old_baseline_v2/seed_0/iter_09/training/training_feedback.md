# Training Feedback

## Final-policy outcome
score=-63.446611, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-93.829840, -28.841668]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_reward | 893.127342 | 38.1% | 38.1% | 100.0% |
| angle_reward | 882.158499 | 37.7% | 37.7% | 100.0% |
| proximity_reward | 556.372235 | 23.8% | 23.8% | 100.0% |
| fuel_penalty | -10.000000 | -0.4% | 0.4% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
