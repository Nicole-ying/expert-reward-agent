# Training Feedback

## Final-policy outcome
score=-105.797799, len=98.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-126.807697, -64.000703]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 110.000000 | 97.8% | 97.8% | 0.6% |
| shaping_reward | 1.218268 | 1.1% | 1.6% | 100.0% |
| contact_bonus | 0.360000 | 0.3% | 0.3% | 2.3% |
| fuel_penalty | -0.322000 | -0.3% | 0.3% | 32.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 16/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
