# Training Feedback

## Final-policy outcome
score=205.709854, len=799.700000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[134.269761, 297.444756]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 175.623698 | 98.1% | 98.1% | 73.9% |
| progress | 2.081827 | 1.2% | 1.2% | 99.3% |
| orientation_penalty | -1.126998 | -0.6% | 0.6% | 100.0% |
| velocity_penalty | -0.182430 | -0.1% | 0.1% | 99.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
