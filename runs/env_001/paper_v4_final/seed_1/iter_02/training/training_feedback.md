# Training Feedback

## Final-policy outcome
score=146.768291, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[115.552230, 182.295756]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_incentive | 230.426260 | 99.3% | 99.3% | 100.0% |
| progress_reward | 1.402875 | 0.6% | 0.6% | 100.0% |
| angle_penalty | -0.121699 | -0.1% | 0.1% | 0.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
