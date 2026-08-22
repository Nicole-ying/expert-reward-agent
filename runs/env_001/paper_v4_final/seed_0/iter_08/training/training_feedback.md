# Training Feedback

## Final-policy outcome
score=180.668432, len=340.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-114.972359, 276.955754]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 26.321993 | 71.7% | 71.7% | 10.1% |
| soft_landing | 5.973221 | 16.3% | 16.3% | 62.1% |
| contact_stability | 3.073930 | 8.4% | 8.4% | 10.3% |
| progress_reward | 1.285477 | 3.5% | 3.6% | 96.4% |
| angle_hinge_penalty | -0.009491 | -0.0% | 0.0% | 4.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
