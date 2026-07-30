# Training Feedback

## Final-policy outcome
score=424.600844, len=872.500000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-361.504219, 759.440175]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1637.012223 | 63.1% | 63.3% | 95.6% |
| _height_gate | 836.270627 | 32.2% | 32.2% | 99.9% |
| lateral_penalty | -113.751870 | -4.4% | 4.4% | 97.1% |
| action_penalty | -2.665006 | -0.1% | 0.1% | 100.0% |
| upright_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
