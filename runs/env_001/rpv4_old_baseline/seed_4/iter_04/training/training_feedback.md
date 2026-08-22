# Training Feedback

## Final-policy outcome
score=-115.832285, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.820937, -91.955470]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| base_contact_bonus | 120.000000 | 42.7% | 42.7% | 1.8% |
| soft_contact_bonus | 105.000000 | 37.4% | 37.4% | 0.5% |
| speed_penalty | -36.639149 | -13.0% | 13.0% | 100.0% |
| angvel_penalty | -9.765215 | -3.5% | 3.5% | 99.6% |
| survival_penalty | -6.840000 | -2.4% | 2.4% | 100.0% |
| approach_reward | 2.236960 | 0.8% | 0.8% | 100.0% |
| angle_penalty | -0.294270 | -0.1% | 0.1% | 100.0% |
| engine_penalty | -0.108000 | -0.0% | 0.0% | 5.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
