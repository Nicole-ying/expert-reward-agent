# Training Feedback

## Final-policy outcome
score=-109.747483, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.896021, -96.173600]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_reward | -197.726848 | -96.5% | 96.5% | 100.0% |
| landing_bounty | 5.815611 | 2.8% | 2.8% | 1.5% |
| angle_penalty | -1.282464 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
