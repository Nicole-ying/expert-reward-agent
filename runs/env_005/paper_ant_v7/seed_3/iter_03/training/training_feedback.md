# Training Feedback

## Final-policy outcome
score=-112.406564, len=369.000000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-913.097532, 121.802362]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_gated | 302.770710 | 84.5% | 92.5% | 70.6% |
| action_penalty | -15.850816 | -4.4% | 4.4% | 100.0% |
| height_reward | -10.963038 | -3.1% | 3.1% | 21.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
