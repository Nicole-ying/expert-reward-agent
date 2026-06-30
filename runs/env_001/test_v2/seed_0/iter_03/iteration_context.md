# Agent Context

- iteration: 3
- target_score: 200.000
- best_score: -110.396 (iter 1)
- current_score: -108.448
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
      "issue": "negative mean, but may be necessary for shaping"
    },
    "progress_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "positive mean but low magnitude relative to penalties"
    },
    "soft_landing_bonus": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "weak",
      "issue": "very low nonzero rate (0.9%), sparse signal"
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "strong",
      "issue": "dominant negative component, mean -0.217, min -1.74"
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
    "best_score_this_skeleton": -108.45,
    "stagnant": false,
    "skeleton_family": "anchor+progress+proxy+constraint"
  },
  "recommended_action": "tune",
  "reasoning": "Stability penalty dominates total reward (mean -0.217 vs progress 0.243), causing negative total. Soft landing bonus is too sparse (0.9% nonzero). External score is very low (-108). Recommend reducing stability penalty coefficient and increasing soft landing bonus frequency or magnitude.",
  "new_lessons": [
    "Stability penalty coefficient must be balanced to avoid dominating progress reward",
    "Soft landing bonus needs higher trigger rate or magnitude to provide useful signal"
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
- Previously tried skeleton family: anchor+progress+proxy+constraint

## Training Feedback (raw evidence)

# Training Feedback

## External evaluation
- score: -108.448347
- episode_length: 74.100000 (mean)
- range: [-120.984102, -97.182870]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.097128 | 0.097128 | 1.000000 | -0.169501 | -0.000119 |
| progress_reward | 0.242793 | 0.256757 | 0.999993 | -0.621055 | 0.636037 |
| soft_landing_bonus | 0.004115 | 0.004115 | 0.009211 | 0.000000 | 0.914699 |
| stability_penalty | -0.216834 | 0.216834 | 1.000000 | -1.739710 | -0.000000 |
| total_reward | -0.067054 | 0.117649 | 1.000000 | -1.895030 | 0.912603 |
| generated_reward | -0.067054 | 0.117649 | 1.000000 | -1.895030 | 0.912603 |
| original_env_reward | -1.570631 | 2.392677 | 1.000000 | -100.000000 | 136.063634 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus; penalty_dominance:stability_penalty; penalty_dominance:original_env_reward
