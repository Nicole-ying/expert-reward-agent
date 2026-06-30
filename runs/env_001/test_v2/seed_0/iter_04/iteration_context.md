# Agent Context

- iteration: 4
- target_score: 200.000
- best_score: -110.396 (iter 1)
- current_score: -111.239
- trend: declining_from_best
- guidance: Investigate why score dropped from best. Consider reverting harmful changes.
- suggested_action: tune or rebuild

The analysis report and expert cards below provide more detailed diagnostic evidence.
Use them to decide your concrete action (tune/add/delete/mix/rebuild).

# Iteration Context for Reward Revision

## Agent Memory (history table)

| iter | score | best | skeleton_summary | trend |
|------|-------|------|------------------|-------|

## Diagnosis Guidance

### Analysis Summary
```json
{
  "failure_modes": [
    "stability_penalty_dominance",
    "sparse_completion_proxy"
  ],
  "hacking_risks": [
    "stability_penalty_dominance",
    "sparse_completion_proxy"
  ],
  "component_analysis": {
    "distance_anchor": {
      "role": "anchor",
      "direction": "negative",
      "signal_strength": "moderate",
      "issue": "negative mean indicates agent is far from target; may need stronger shaping"
    },
    "progress_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "positive mean but external score is very negative; progress not translating to success"
    },
    "soft_landing_bonus": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "weak",
      "issue": "very low nonzero rate (0.9%); sparse and rarely triggered"
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "strong",
      "issue": "large negative mean (-0.218) and always active; dominates total reward"
    }
  },
  "skeleton_assessment": {
    "current_skeleton": [
      "distance_anchor",
      "progress_reward",
      "soft_landing_bonus",
      "stability_penalty"
    ],
    "iterations_on_this_skeleton": 1,
    "best_score_this_skeleton": -111.24,
    "stagnant": false,
    "skeleton_family": "anchor+progress+proxy+constraint"
  },
  "recommended_action": "tune",
  "reasoning": "外部得分-111.24远低于目标200，且original_env_reward均值-1.53表明环境惩罚大。stability_penalty均值-0.218且始终激活，主导总奖励，符合stability_penalty_dominance。soft_landing_bonus触发率仅0.9%，稀疏且贡献小，符合sparse_completion_proxy。progress_reward正向但未转化为成功，需调整系数。建议微调：降低stability_penalty系数，提高progress_reward系数，并增加soft_landing_bonus触发条件。",
  "new_lessons": [
    "stability_penalty coefficient should be reduced to avoid dominating total reward",
    "soft_landing_bonus needs more frequent triggering to provide useful shaping"
  ]
}
```

### Expert Cards (compressed)
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target
## sparse_completion_proxy
- signal: completion/landing bonus trigger_rate < 1%
- risk: final bonus provides little early learning guidance
- fix: replace hard bonus with smoother landing-quality shaping

### KB Recommended Skeletons for task `navigation_goal_reaching`
- time_penalty, distance_reward, progress_delta_reward, potential_based_shaping, gated_reward
- Previously tried skeleton family: anchor+progress+proxy+constraint

## Training Feedback (raw evidence)

# Training Feedback

## External evaluation
- score: -111.238907
- episode_length: 74.100000 (mean)
- range: [-120.969771, -105.485038]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.096984 | 0.096984 | 1.000000 | -0.170105 | -0.000176 |
| progress_reward | 0.323339 | 0.341766 | 0.999995 | -0.807268 | 0.844599 |
| soft_landing_bonus | 0.003982 | 0.003982 | 0.009369 | 0.000000 | 0.914458 |
| stability_penalty | -0.218167 | 0.218167 | 1.000000 | -1.570842 | -0.000001 |
| total_reward | 0.012169 | 0.155207 | 1.000000 | -2.001005 | 0.919099 |
| generated_reward | 0.012169 | 0.155207 | 1.000000 | -2.001005 | 0.919099 |
| original_env_reward | -1.526222 | 2.404256 | 1.000000 | -100.000000 | 129.532942 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus; penalty_dominance:original_env_reward
