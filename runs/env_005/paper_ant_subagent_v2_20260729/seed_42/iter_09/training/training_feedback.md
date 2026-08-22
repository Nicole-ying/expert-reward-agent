# Training Feedback

## Final-policy outcome
score=-375.549769, len=662.050000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-1204.734137, 50.325056]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 2742.435287 | 82.1% | 90.5% | 100.0% |
| upright_penalty | -252.188203 | -7.6% | 7.6% | 50.7% |
| joint_vel_penalty | -28.667485 | -0.9% | 0.9% | 97.4% |
| lateral_penalty | -27.463394 | -0.8% | 0.8% | 42.4% |
| action_penalty | -3.162914 | -0.1% | 0.1% | 100.0% |
| pitch_vel_penalty | -3.015887 | -0.1% | 0.1% | 5.4% |
| descent_penalty | -1.141368 | -0.0% | 0.0% | 26.9% |
| roll_vel_penalty | -0.409518 | -0.0% | 0.0% | 1.1% |
| ascent_penalty | -0.165317 | -0.0% | 0.0% | 1.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
