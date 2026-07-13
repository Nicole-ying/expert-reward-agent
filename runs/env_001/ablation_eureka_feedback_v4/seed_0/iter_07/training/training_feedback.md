# Training Feedback

## Final-policy outcome
score=145.215369, len=836.050000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[72.472046, 248.304170]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability | 830.748713 | 66.7% | 66.7% | 100.0% |
| contact | 407.966268 | 32.7% | 32.7% | 46.6% |
| progress | 7.051247 | 0.6% | 0.6% | 77.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
