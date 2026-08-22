# Training Feedback

## Final-policy outcome
score=186.237167, len=797.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-42.514719, 306.229817]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 2399.256817 | 83.6% | 83.6% | 99.7% |
| gait_quality | 340.200000 | 11.9% | 11.9% | 42.7% |
| gait_efficiency | 82.693568 | 2.9% | 2.9% | 42.4% |
| both_off_penalty | -29.995000 | -1.0% | 1.0% | 37.6% |
| action_penalty | -12.450501 | -0.4% | 0.4% | 100.0% |
| balance_penalty | -5.856184 | -0.2% | 0.2% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
