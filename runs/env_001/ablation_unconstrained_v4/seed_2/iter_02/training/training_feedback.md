# Training Feedback

## Final-policy outcome
score=-21.247497, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-51.407924, 18.199478]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| horizontal_penalty | -10.761334 | -86.7% | 86.7% | 100.0% |
| progress | 1.189368 | 9.6% | 11.9% | 100.0% |
| att_penalty | -0.089810 | -0.7% | 0.7% | 3.2% |
| vel_penalty | -0.085200 | -0.7% | 0.7% | 1.1% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
