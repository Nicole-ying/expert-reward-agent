# Training Feedback

## Final-policy outcome
score=-113.486473, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.582396, -95.638888]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 40.000000 | 77.7% | 77.7% | 0.6% |
| shaping | 5.144894 | 10.0% | 16.2% | 100.0% |
| step_penalty | -1.368000 | -2.7% | 2.7% | 100.0% |
| angle_penalty | -1.198616 | -2.3% | 2.3% | 100.0% |
| contact_continuous | 0.277654 | 0.5% | 0.5% | 0.9% |
| angvel_penalty | -0.227795 | -0.4% | 0.4% | 100.0% |
| fuel_penalty | -0.064000 | -0.1% | 0.1% | 4.7% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
