# Training Feedback

## Final-policy outcome
score=120.839410, len=291.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-104.546529, 267.530685]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 2627.500000 | 97.4% | 97.4% | 18.0% |
| contact_reward | 59.275000 | 2.2% | 2.2% | 22.1% |
| action_reward | -5.524000 | -0.2% | 0.2% | 52.1% |
| vel_damping | -2.258444 | -0.1% | 0.1% | 37.0% |
| progress_reward | 1.721344 | 0.1% | 0.1% | 97.8% |
| orientation_penalty | -0.798858 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
