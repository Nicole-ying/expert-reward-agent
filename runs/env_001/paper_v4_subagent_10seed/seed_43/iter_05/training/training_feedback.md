# Training Feedback

## Final-policy outcome
score=-55.775241, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-84.568107, -19.444417]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_approach_reward | 431.991723 | 98.1% | 98.1% | 100.0% |
| progress | 7.347618 | 1.7% | 1.9% | 100.0% |
| contact_success_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
