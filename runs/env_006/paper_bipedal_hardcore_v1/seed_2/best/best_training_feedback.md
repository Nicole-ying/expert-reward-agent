# Training Feedback

## Final-policy outcome
score=-52.194184, len=401.700000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-97.732651, 30.117148]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 71.543493 | 94.5% | 94.5% | 78.8% |
| energy_penalty | -3.699793 | -4.9% | 4.9% | 100.0% |
| hinge_penalty | -0.447545 | -0.6% | 0.6% | 2.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
