# Training Feedback

## Final-policy outcome
score=1414.466833, len=702.900000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[13.191050, 2368.687431]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 6466.046076 | 94.6% | 94.7% | 100.0% |
| height_reward | 208.185373 | 3.0% | 3.0% | 100.0% |
| action_cost | -155.394325 | -2.3% | 2.3% | 100.0% |
| height_boundary_penalty | -0.316544 | -0.0% | 0.0% | 0.3% |
| upright_penalty | -0.270802 | -0.0% | 0.0% | 0.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
