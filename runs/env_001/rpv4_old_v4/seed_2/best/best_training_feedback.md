# Training Feedback

## Final-policy outcome
score=237.757371, len=517.050000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[135.690384, 300.934723]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 179.747522 | 98.9% | 98.9% | 60.9% |
| orientation_penalty | -1.134041 | -0.6% | 0.6% | 100.0% |
| progress | 0.685672 | 0.4% | 0.4% | 98.2% |
| velocity_penalty | -0.183836 | -0.1% | 0.1% | 97.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
