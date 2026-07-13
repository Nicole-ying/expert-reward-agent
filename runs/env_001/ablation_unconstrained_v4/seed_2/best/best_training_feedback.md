# Training Feedback

## Final-policy outcome
score=71.060291, len=977.600000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[17.969578, 176.793787]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 633.930169 | 42.8% | 42.8% | 26.0% |
| alignment | 479.401371 | 32.3% | 32.3% | 100.0% |
| height_near_reward | 356.299293 | 24.0% | 24.0% | 85.7% |
| progress | 6.293330 | 0.4% | 0.6% | 100.0% |
| y_progress | 2.818477 | 0.2% | 0.2% | 93.1% |
| horizontal_penalty | -1.001960 | -0.1% | 0.1% | 100.0% |
| vel_penalty | -0.023368 | -0.0% | 0.0% | 1.4% |
| att_penalty | -0.014444 | -0.0% | 0.0% | 2.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
