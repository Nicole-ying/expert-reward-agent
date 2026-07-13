# Training Feedback

## Final-policy outcome
score=-8.161178, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-32.043324, 26.803647]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| alignment | 493.779207 | 95.7% | 95.7% | 100.0% |
| y_progress | 13.406476 | 2.6% | 2.9% | 100.0% |
| progress | 6.624235 | 1.3% | 1.5% | 100.0% |
| vel_penalty | -0.019312 | -0.0% | 0.0% | 1.3% |
| att_penalty | -0.004685 | -0.0% | 0.0% | 1.6% |
| any_contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| height_near_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| stable_contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
