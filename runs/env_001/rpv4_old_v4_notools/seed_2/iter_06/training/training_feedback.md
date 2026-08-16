# Training Feedback

## Final-policy outcome
score=167.898467, len=495.400000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-19.888215, 258.940882]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_contact_bonus | 339.500000 | 68.3% | 68.3% | 34.3% |
| contact_bonus | 148.500000 | 29.9% | 29.9% | 5.6% |
| progress | 3.963728 | 0.8% | 1.0% | 99.8% |
| orientation_penalty | -2.458427 | -0.5% | 0.5% | 100.0% |
| velocity_penalty | -1.288687 | -0.3% | 0.3% | 99.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
