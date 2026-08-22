# Training Feedback

## Final-policy outcome
score=-211.111008, len=659.450000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-270.754536, -109.435024]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 540.673710 | 82.5% | 82.5% | 100.0% |
| landing_quality | 76.275135 | 11.6% | 11.6% | 100.0% |
| engine_cost | -37.086500 | -5.7% | 5.7% | 91.0% |
| descent_reward | 1.497447 | 0.2% | 0.2% | 50.5% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
