# Training Feedback

## Final-policy outcome
score=-24.783035, len=143.150000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-106.555213, 19.614195]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_gated | 30.247016 | 64.5% | 75.8% | 100.0% |
| contact_landing_reward | 9.951514 | 21.2% | 21.2% | 4.6% |
| lateral_pos_penalty | -1.106193 | -2.4% | 2.4% | 100.0% |
| angvel_penalty | -0.294459 | -0.6% | 0.6% | 99.9% |
| angle_term_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
