# Training Feedback

## Final-policy outcome
score=-139.005476, len=536.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-211.556638, 126.303479]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_near_reward | 71.094141 | 64.3% | 64.3% | 69.6% |
| alignment | 19.234215 | 17.4% | 17.4% | 69.6% |
| any_contact_reward | 7.500000 | 6.8% | 6.8% | 0.7% |
| y_progress | 5.759473 | 5.2% | 5.5% | 100.0% |
| horizontal_penalty | -3.611519 | -3.3% | 3.3% | 100.0% |
| progress | 0.792169 | 0.7% | 2.7% | 100.0% |
| att_penalty | -0.010061 | -0.0% | 0.0% | 3.5% |
| vel_penalty | -0.008770 | -0.0% | 0.0% | 2.6% |
| stable_contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
