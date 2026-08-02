# Training Feedback

## Final-policy outcome
score=142.739469, len=885.150000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[50.795260, 241.661744]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 306.275000 | 50.2% | 50.2% | 69.2% |
| contact_reward | 285.840011 | 46.8% | 46.8% | 69.5% |
| progress_delta | 10.098140 | 1.7% | 1.9% | 99.9% |
| boundary_warning | -5.465823 | -0.9% | 0.9% | 4.6% |
| angle_penalty | -0.440156 | -0.1% | 0.1% | 4.1% |
| speed_penalty | -0.235103 | -0.0% | 0.0% | 3.2% |
| angvel_penalty | -0.070553 | -0.0% | 0.0% | 0.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
