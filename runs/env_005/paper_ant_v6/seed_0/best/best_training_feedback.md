# Training Feedback

## Final-policy outcome
score=18.414726, len=20.850000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-14.424808, 51.287824]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 19.870009 | 62.6% | 62.6% | 73.9% |
| lateral_penalty | -5.775045 | -18.2% | 18.2% | 100.0% |
| action_penalty | -5.647933 | -17.8% | 17.8% | 100.0% |
| height_penalty | -0.453926 | -1.4% | 1.4% | 4.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
