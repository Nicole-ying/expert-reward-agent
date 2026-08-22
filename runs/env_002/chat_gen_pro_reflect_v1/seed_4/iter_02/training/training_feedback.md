# Training Feedback

## Final-policy outcome
score=287.885100, len=901.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[130.283291, 297.257986]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_factor | 1716.583954 | 54.2% | 54.2% | 100.0% |
| gated_forward | 949.184075 | 30.0% | 30.0% | 100.0% |
| forward_reward | 498.291245 | 15.7% | 15.7% | 100.0% |
| balance_penalty | -0.142115 | -0.0% | 0.0% | 100.0% |
| angle_penalty | -0.089531 | -0.0% | 0.0% | 0.0% |
| angular_vel_penalty | -0.052584 | -0.0% | 0.0% | 100.0% |
| angle_deviation | 0.022383 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
