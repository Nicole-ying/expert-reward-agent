# Training Feedback

## Final-policy outcome
score=-383.092850, len=503.550000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-1658.186499, 9.208281]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward | 329.744891 | 42.5% | 48.1% | 78.5% |
| upright_penalty | -367.131018 | -47.4% | 47.4% | 100.0% |
| lateral_penalty | -32.104537 | -4.1% | 4.1% | 78.1% |
| height_penalty | -1.570144 | -0.2% | 0.2% | 41.0% |
| action_penalty | -1.519467 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
