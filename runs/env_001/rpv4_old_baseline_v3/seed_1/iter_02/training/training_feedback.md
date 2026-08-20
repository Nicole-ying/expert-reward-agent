# Training Feedback

## Final-policy outcome
score=-113.739688, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-136.258679, -93.458690]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_penalty | -10.987555 | -20.8% | 20.8% | 100.0% |
| attitude_penalty | -9.715681 | -18.4% | 18.4% | 100.0% |
| position_quality | 8.862232 | 16.7% | 16.7% | 100.0% |
| approach | 8.487203 | 16.0% | 16.6% | 100.0% |
| step_cost | -6.840000 | -12.9% | 12.9% | 100.0% |
| contact_reward | 4.055390 | 7.7% | 7.7% | 3.1% |
| hard_landing | -3.589825 | -6.8% | 6.8% | 1.5% |
| fuel_cost | -0.109500 | -0.2% | 0.2% | 5.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
