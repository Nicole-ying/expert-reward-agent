# Training Feedback

## Final-policy outcome
score=-46.323659, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-77.088527, -8.231464]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity | 307.998292 | 97.2% | 97.2% | 100.0% |
| progress | 4.670757 | 1.5% | 1.7% | 100.0% |
| vel_penalty | -1.862840 | -0.6% | 0.6% | 100.0% |
| angle_penalty | -1.468444 | -0.5% | 0.5% | 100.0% |
| angvel_penalty | -0.031268 | -0.0% | 0.0% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
