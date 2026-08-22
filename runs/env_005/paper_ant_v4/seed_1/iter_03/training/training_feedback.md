# Training Feedback

## Final-policy outcome
score=265.019931, len=852.650000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[8.752045, 591.017548]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| upright_stability | 750.874398 | 44.1% | 49.1% | 100.0% |
| forward_progress | 572.500129 | 33.6% | 48.6% | 51.8% |
| height_penalty | -37.662677 | -2.2% | 2.2% | 5.5% |
| lateral_drift_penalty | -1.084919 | -0.1% | 0.1% | 28.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
