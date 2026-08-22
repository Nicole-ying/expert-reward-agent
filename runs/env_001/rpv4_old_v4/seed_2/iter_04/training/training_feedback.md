# Training Feedback

## Final-policy outcome
score=-52.526070, len=71.850000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-89.558714, -24.848525]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 5.533849 | 82.2% | 85.7% | 100.0% |
| velocity_penalty | -0.467862 | -6.9% | 6.9% | 99.9% |
| contact_guidance | 0.252135 | 3.7% | 3.7% | 0.8% |
| orientation_penalty | -0.242670 | -3.6% | 3.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 10/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
