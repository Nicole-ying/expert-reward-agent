# Training Feedback

## Final-policy outcome
score=-61.568834, len=303.450000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-104.995859, -26.032110]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| posture_gate | 200.961421 | 92.0% | 92.0% | 100.0% |
| progress_reward | 17.012064 | 7.8% | 7.9% | 99.1% |
| vertical_penalty | -0.109440 | -0.1% | 0.1% | 77.7% |
| angular_penalty | -0.021244 | -0.0% | 0.0% | 77.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
