# Training Feedback

## Final-policy outcome
score=-108.051765, len=509.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-159.311185, 144.665415]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability | 500.565051 | 97.0% | 97.0% | 100.0% |
| engine_penalty | -10.558500 | -2.0% | 2.0% | 69.1% |
| contact | 4.536741 | 0.9% | 0.9% | 1.5% |
| progress | 0.188374 | 0.0% | 0.0% | 40.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
