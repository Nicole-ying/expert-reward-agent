# Training Feedback

## Final-policy outcome
score=-13.379476, len=936.950000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-62.123600, 145.329135]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| lateral_pos_penalty | -57.107624 | -47.5% | 47.5% | 100.0% |
| progress_gated | 27.726811 | 23.1% | 34.7% | 100.0% |
| contact_landing_reward | 21.232197 | 17.7% | 17.7% | 0.8% |
| angvel_penalty | -0.079490 | -0.1% | 0.1% | 99.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
