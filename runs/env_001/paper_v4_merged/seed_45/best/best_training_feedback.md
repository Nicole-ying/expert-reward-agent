# Training Feedback

## Final-policy outcome
score=195.264779, len=754.650000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[123.823677, 273.594493]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| completion_proxy | 466.267267 | 98.3% | 98.3% | 64.5% |
| progress_delta | 6.614631 | 1.4% | 1.5% | 99.1% |
| speed_penalty | -0.394443 | -0.1% | 0.1% | 4.2% |
| angle_penalty | -0.219735 | -0.0% | 0.0% | 3.2% |
| angvel_penalty | -0.042918 | -0.0% | 0.0% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
