# Training Feedback

## Final-policy outcome
score=-61.551621, len=253.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-108.189426, -33.623886]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 24.028233 | 91.0% | 96.8% | 100.0% |
| vertical_penalty | -0.483851 | -1.8% | 1.8% | 100.0% |
| posture_penalty | -0.311302 | -1.2% | 1.2% | 3.9% |
| angular_penalty | -0.053693 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
