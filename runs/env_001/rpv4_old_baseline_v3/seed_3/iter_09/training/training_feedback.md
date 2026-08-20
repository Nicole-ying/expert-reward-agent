# Training Feedback

## Final-policy outcome
score=-222.793041, len=891.150000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-304.958100, -136.051119]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 889.755093 | 84.7% | 84.7% | 100.0% |
| speed_penalty | -67.352662 | -6.4% | 6.4% | 100.0% |
| time_penalty | -44.557500 | -4.2% | 4.2% | 100.0% |
| fuel_cost | -26.734500 | -2.5% | 2.5% | 100.0% |
| orientation_penalty | -20.100695 | -1.9% | 1.9% | 100.0% |
| progress_reward | -0.924371 | -0.1% | 0.2% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| touchdown_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
