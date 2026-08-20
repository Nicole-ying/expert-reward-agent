# Training Feedback

## Final-policy outcome
score=158.182209, len=477.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-51.101341, 230.659373]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 464.901036 | 94.0% | 94.0% | 89.4% |
| fuel_penalty | -22.560000 | -4.6% | 4.6% | 94.5% |
| progress | 5.764077 | 1.2% | 1.3% | 98.0% |
| orientation_penalty | -0.855178 | -0.2% | 0.2% | 73.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
