# Training Feedback

## Final-policy outcome
score=81.184393, len=890.750000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-27.608441, 204.846063]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_completion | 478.927441 | 84.8% | 84.8% | 15.8% |
| soft_landing_speed_penalty | -73.990109 | -13.1% | 13.1% | 100.0% |
| approach_progress | 5.892979 | 1.0% | 1.2% | 100.0% |
| orientation_penalty | -5.213019 | -0.9% | 0.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
