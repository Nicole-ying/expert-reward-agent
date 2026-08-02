# Training Feedback

## Final-policy outcome
score=-2081.095022, len=696.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-4566.700406, -1319.707065]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gate_angle | 313.229992 | 41.8% | 41.8% | 100.0% |
| contact_factor | 278.925000 | 37.2% | 37.2% | 100.0% |
| progress_bonus | 100.462904 | 13.4% | 13.4% | 35.2% |
| progress | -5.138309 | -0.7% | 6.0% | 100.0% |
| shaped_progress | -2.464002 | -0.3% | 1.0% | 100.0% |
| action_cost | -3.195500 | -0.4% | 0.4% | 45.9% |
| speed_penalty | -1.172610 | -0.2% | 0.2% | 0.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
