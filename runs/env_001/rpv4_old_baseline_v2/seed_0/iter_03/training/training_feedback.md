# Training Feedback

## Final-policy outcome
score=-109.411569, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.079360, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vel_penalty | -0.915253 | -36.9% | 36.9% | 100.0% |
| landing_reward | 0.831775 | 33.5% | 33.5% | 3.1% |
| progress_reward | 0.542858 | 21.9% | 24.5% | 100.0% |
| fuel_penalty | -0.076000 | -3.1% | 3.1% | 5.6% |
| ang_penalty | -0.051761 | -2.1% | 2.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
