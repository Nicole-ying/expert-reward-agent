# Training Feedback

## Final-policy outcome
score=59.178993, len=764.050000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-63.062786, 245.589857]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 298.305334 | 87.0% | 87.0% | 100.0% |
| engine_penalty | -22.212000 | -6.5% | 6.5% | 96.9% |
| contact_reward | 20.250000 | 5.9% | 5.9% | 2.7% |
| progress_reward | 1.908250 | 0.6% | 0.6% | 56.4% |
| upright_penalty | -0.138744 | -0.0% | 0.0% | 3.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
