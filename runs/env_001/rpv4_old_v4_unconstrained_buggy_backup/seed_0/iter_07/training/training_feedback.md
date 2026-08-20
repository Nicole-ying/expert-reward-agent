# Training Feedback

## Final-policy outcome
score=-123.966105, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-148.869029, -104.426353]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 13.368587 | 46.0% | 47.6% | 100.0% |
| vel_damping | -7.139505 | -24.6% | 24.6% | 100.0% |
| landing_bonus | 6.537704 | 22.5% | 22.5% | 3.3% |
| orientation_penalty | -1.407758 | -4.8% | 4.8% | 100.0% |
| action_penalty | -0.120000 | -0.4% | 0.4% | 3.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
