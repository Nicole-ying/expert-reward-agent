# Training Feedback

## Final-policy outcome
score=11.665668, len=602.850000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-207.933331, 257.869138]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 1131.888208 | 97.9% | 97.9% | 6.6% |
| progress | 9.127868 | 0.8% | 1.1% | 99.4% |
| fuel_penalty | -10.440000 | -0.9% | 0.9% | 86.6% |
| speed_penalty | -0.272128 | -0.0% | 0.0% | 2.1% |
| angle_penalty | -0.063367 | -0.0% | 0.0% | 0.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
