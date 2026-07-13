# Training Feedback

## Final-policy outcome
score=21.264371, len=771.050000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[-348.340888, 236.013030]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| touchdown_success | 494.812610 | 84.0% | 84.0% | 8.4% |
| landing_quality | 55.679955 | 9.5% | 9.5% | 100.0% |
| velocity_penalty | -21.543678 | -3.7% | 3.7% | 18.8% |
| angle_stability | -9.542428 | -1.6% | 1.6% | 100.0% |
| approach_progress | 5.917686 | 1.0% | 1.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
