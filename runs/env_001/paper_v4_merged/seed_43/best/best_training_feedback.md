# Training Feedback

## Final-policy outcome
score=-24.047127, len=980.750000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-127.219130, 24.581447]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_angle | 916.943191 | 65.1% | 65.1% | 100.0% |
| contact_factor | 398.390000 | 28.3% | 28.3% | 100.0% |
| success_bonus | 75.372604 | 5.4% | 5.4% | 100.0% |
| action_cost | -9.765000 | -0.7% | 0.7% | 99.6% |
| speed_penalty | -5.777554 | -0.4% | 0.4% | 77.6% |
| progress | 1.191460 | 0.1% | 0.1% | 100.0% |
| shaped_progress | 0.410168 | 0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
