# Agent Context

- iteration: 2
- target_score: 200.000
- best_score: -110.396 (iter 1)
- current_score: -110.396
- trend: searching
- guidance: Continue refining based on evidence.
- suggested_action: tune or add

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
    "early_failure_or_crash"
  ],
  "hacking_risks": [
    "stability_penalty_dominance"
  ],
  "component_analysis": {
    "distance_anchor": {
      "role": "anchor",
      "direction": "negative",
      "signal_strength": "moderate",
      "issue": "negative mean indicates agent is far from target, but magnitude is small"
    },
    "progress_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "positive mean but small magnitude relative to penalties"
    },
    "soft_landing_bonus": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "weak",
      "issue": "very low nonzero rate (0.5%), sparse and ineffective"
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "strong",
      "issue": "dominant negative component with large magnitude, overwhelming other signals"
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
    "best_score_this_skeleton": -110.396465,
    "stagnant": false,
    "skeleton_family": "progress+stability+landing_proxy+anchor"
  },
  "recommended_action": "tune",
  "reasoning": "Stability penalty dominates (mean -0.343, strong negative), causing total reward negative. Progress reward positive but weak. Soft landing bonus too sparse. External score very low (-110) with early failures. Recommend reducing stability penalty coefficient and increasing progress reward coefficient to balance.",
  "new_lessons": [
    "Stability penalty coefficient must be reduced to avoid penalty dominance",
    "Soft landing bonus needs higher nonzero rate or alternative shaping"
  ]
}
```

### Expert Cards (compressed)
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target
## early_failure_or_crash
- signal: negative external score and short episode length
- risk: reward does not guide stable control before termination
- fix: add smooth approach/landing signals; avoid relying on sparse terminal-like proxy

### KB Recommended Skeletons for task `navigation_goal_reaching`
- time_penalty, distance_reward, progress_delta_reward, potential_based_shaping, gated_reward
- Previously tried skeleton family: progress+stability+landing_proxy+anchor

## Training Feedback (raw evidence)

# Training Feedback

## External evaluation
- score: -110.396465
- episode_length: 74.100000 (mean)
- range: [-121.667740, -97.714115]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.097176 | 0.097176 | 1.000000 | -0.169501 | -0.000123 |
| progress_reward | 0.160700 | 0.170018 | 0.999987 | -0.416105 | 0.424384 |
| soft_landing_bonus | 0.009905 | 0.009905 | 0.004953 | 0.000000 | 2.000000 |
| stability_penalty | -0.342913 | 0.342913 | 1.000000 | -2.804435 | -0.000000 |
| total_reward | -0.269484 | 0.288008 | 1.000000 | -3.049366 | 1.996958 |
| generated_reward | -0.269484 | 0.288008 | 1.000000 | -3.049366 | 1.996958 |
| original_env_reward | -1.562164 | 2.397884 | 1.000000 | -100.000000 | 131.099741 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus; penalty_dominance:stability_penalty; penalty_dominance:generated_reward; penalty_dominance:original_env_reward
