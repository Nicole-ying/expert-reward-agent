# Training Feedback

## Final-policy outcome
score=8.916904, len=17.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-16.469643, 35.759458]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 15.924914 | 61.7% | 61.7% | 69.3% |
| action_penalty | -5.568962 | -21.6% | 21.6% | 100.0% |
| lateral_penalty | -4.312249 | -16.7% | 16.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
