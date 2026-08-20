# Training Feedback

## Final-policy outcome
score=1647.491412, len=656.050000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[295.609801, 2630.356416]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 993.442217 | 98.9% | 98.9% | 100.0% |
| ang_vel_penalty | -11.534522 | -1.1% | 1.1% | 100.0% |
| posture_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
