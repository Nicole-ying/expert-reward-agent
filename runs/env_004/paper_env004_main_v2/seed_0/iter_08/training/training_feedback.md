# Training Feedback

## Final-policy outcome
score=2846.231479, len=963.950000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[2213.218937, 3028.645702]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_posture_reward | 1850.257267 | 99.3% | 99.3% | 100.0% |
| ang_vel_penalty | -13.739192 | -0.7% | 0.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
