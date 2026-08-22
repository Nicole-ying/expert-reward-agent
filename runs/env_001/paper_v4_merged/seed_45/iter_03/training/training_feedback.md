# Training Feedback

## Final-policy outcome
score=141.281395, len=897.700000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[19.392306, 265.552154]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 108.335002 | 59.9% | 59.9% | 70.9% |
| landing_progress | 70.443132 | 39.0% | 39.0% | 70.4% |
| progress_delta | 1.347342 | 0.7% | 0.8% | 100.0% |
| speed_penalty | -0.333577 | -0.2% | 0.2% | 4.8% |
| orientation_penalty | -0.129582 | -0.1% | 0.1% | 2.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
