# Training Feedback

## Final-policy outcome
score=138.805331, len=955.300000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[85.921617, 277.205748]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 1507.757777 | 63.9% | 63.9% | 3.3% |
| settled_reward | 795.894961 | 33.7% | 33.7% | 32.4% |
| landing_approach | 30.274994 | 1.3% | 1.3% | 100.0% |
| goal_proximity | -11.122742 | -0.5% | 0.5% | 100.0% |
| fuel_penalty | -9.113000 | -0.4% | 0.4% | 95.4% |
| velocity_penalty | -4.275016 | -0.2% | 0.2% | 100.0% |
| orientation_penalty | -1.241583 | -0.1% | 0.1% | 100.0% |
| contact_speed_penalty | -1.037298 | -0.0% | 0.0% | 32.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
