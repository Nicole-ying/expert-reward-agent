# Training Feedback

## Final-policy outcome
score=-65.156679, len=190.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-87.190721, -48.858708]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 130.956489 | 91.9% | 92.7% | 100.0% |
| posture_penalty | -7.892762 | -5.5% | 5.5% | 10.7% |
| action_cost | -2.528537 | -1.8% | 1.8% | 100.0% |
| ang_vel_penalty | -0.013628 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 7/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
