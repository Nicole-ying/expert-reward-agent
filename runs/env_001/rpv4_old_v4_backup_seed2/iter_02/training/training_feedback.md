# Training Feedback

## Final-policy outcome
score=-37.031320, len=962.550000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-119.359794, 188.512930]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing | 697.436041 | 93.8% | 93.8% | 99.6% |
| speed_penalty | -42.129938 | -5.7% | 5.7% | 100.0% |
| angular_vel_penalty | -2.456624 | -0.3% | 0.3% | 100.0% |
| progress | 0.664903 | 0.1% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
