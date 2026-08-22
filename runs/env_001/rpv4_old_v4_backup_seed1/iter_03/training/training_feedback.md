# Training Feedback

## Final-policy outcome
score=21.452647, len=864.550000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-121.813880, 228.013400]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 1011.467861 | 97.7% | 97.7% | 8.2% |
| upright_penalty | -11.451934 | -1.1% | 1.1% | 100.0% |
| velocity_penalty | -9.103126 | -0.9% | 0.9% | 99.9% |
| goal_proximity | 1.768764 | 0.2% | 0.3% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
