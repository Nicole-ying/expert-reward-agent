# Training Feedback

## Final-policy outcome
score=-134.181929, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-166.587315, -84.184200]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity | 13.783368 | 53.2% | 53.2% | 100.0% |
| progress | 0.891089 | 3.4% | 29.7% | 100.0% |
| speed_penalty | -3.393717 | -13.1% | 13.1% | 100.0% |
| engine_penalty | -0.998600 | -3.9% | 3.9% | 99.9% |
| orientation | -0.037038 | -0.1% | 0.1% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
