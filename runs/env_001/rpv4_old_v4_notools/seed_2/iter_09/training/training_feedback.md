# Training Feedback

## Final-policy outcome
score=128.161049, len=841.600000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[24.893681, 242.229440]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_landing_bonus | 494.000000 | 71.4% | 71.4% | 5.9% |
| stable_contact_bonus | 182.325000 | 26.3% | 26.3% | 43.3% |
| progress | 10.235151 | 1.5% | 1.8% | 99.2% |
| orientation_penalty | -2.245467 | -0.3% | 0.3% | 100.0% |
| velocity_penalty | -0.959756 | -0.1% | 0.1% | 99.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
