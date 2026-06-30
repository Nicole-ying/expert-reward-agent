# Agent Context

- iteration: 6
- target_score: 200.000
- best_score: -110.396 (iter 1)
- current_score: -111.139
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
      "issue": "negative mean indicates agent is not staying near target; may be too weak to counteract stability penalty"
    },
    "progress_delta_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "positive mean but insufficient to overcome negative components"
    },
    "soft_landing_bonus": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "weak",
      "issue": "very low nonzero rate (0.46%) indicates sparse reward; agent rarely achieves soft landing"
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "strong",
      "issue": "dominant negative component with large magnitude; likely causing agent to avoid movement"
    }
  },
  "skeleton_assessment": {
    "current_skeleton": [
      "distance_anchor",
      "progress_delta_reward",
      "soft_landing_bonus",
      "stability_penalty"
    ],
    "iterations_on_this_skeleton": 1,
    "best_score_this_skeleton": -111.139031,
    "stagnant": false,
    "skeleton_family": "anchor+progress+proxy+constraint"
  },
  "recommended_action": "tune",
  "reasoning": "Stability penalty dominates (mean -0.341, strong negative), causing agent to stay still to avoid penalty. Soft landing bonus is too sparse (0.46% nonzero rate) to guide learning. Progress reward is positive but insufficient. Recommend reducing stability penalty coefficient and increasing soft landing bonus frequency or magnitude.",
  "new_lessons": [
    "stability_penalty coefficient must be reduced to avoid dominance",
    "soft_landing_bonus needs denser shaping or higher reward to be effective"
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
- score: -111.139031
- episode_length: 74.100000 (mean)
- range: [-122.534771, -105.145711]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.097016 | 0.097016 | 1.000000 | -0.169501 | -0.000019 |
| progress_delta_reward | 0.160916 | 0.170245 | 0.999990 | -0.418539 | 0.425586 |
| soft_landing_bonus | 0.009273 | 0.009273 | 0.004637 | 0.000000 | 2.000000 |
| stability_penalty | -0.340556 | 0.340556 | 1.000000 | -2.804435 | -0.000000 |
| total_reward | -0.267383 | 0.284934 | 1.000000 | -3.049366 | 1.993564 |
| generated_reward | -0.267383 | 0.284934 | 1.000000 | -3.049366 | 1.993564 |
| original_env_reward | -1.538081 | 2.403107 | 1.000000 | -100.000000 | 126.195419 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus
