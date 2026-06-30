# Agent Context

- iteration: 9
- target_score: 200.000
- best_score: 188.212 (iter 8)
- current_score: 188.212
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
    "goal_near_oscillation",
    "high_reward_without_success"
  ],
  "hacking_risks": [
    "goal_near_oscillation",
    "high_reward_without_success"
  ],
  "component_analysis": {
    "distance_anchor": {
      "role": "anchor",
      "direction": "negative",
      "signal_strength": "moderate",
      "issue": "Always negative, may be too strong or misaligned with progress."
    },
    "landing_quality_shaping": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "strong",
      "issue": "High mean but nonzero rate 71%, may encourage landing without sufficient progress."
    },
    "progress_delta_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "Mean positive but min negative, may be insufficient to drive consistent progress."
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "moderate",
      "issue": "Always negative, may penalize necessary maneuvers."
    }
  },
  "skeleton_assessment": {
    "current_skeleton": [
      "distance_anchor",
      "landing_quality_shaping",
      "progress_delta_reward",
      "stability_penalty"
    ],
    "iterations_on_this_skeleton": 1,
    "best_score_this_skeleton": 188.21,
    "stagnant": false,
    "skeleton_family": "anchor+proxy+progress+constraint"
  },
  "recommended_action": "tune",
  "reasoning": "Score 188.21 close to target 200, but external score variance high (min -134). Progress_delta_reward mean 0.24 is low relative to landing_quality_shaping 1.97, suggesting agent may hover near goal without committing to landing. Distance_anchor always negative may pull agent away. Recommend tuning coefficients: increase progress_delta_reward coefficient, decrease landing_quality_shaping and distance_anchor magnitude.",
  "new_lessons": [
    "progress_delta_reward coefficient should be increased to drive consistent progress",
    "landing_quality_shaping may cause premature landing attempts without sufficient progress"
  ]
}
```

### Expert Cards (compressed)
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward
## high_reward_without_success
- signal: generated_reward improves but external score stays poor
- risk: policy optimizes the custom reward but not the real task
- fix: reduce exploitable terms; add constraints tied to actual task progress and stable outcome

### KB Recommended Skeletons for task `navigation_goal_reaching`
- time_penalty, distance_reward, progress_delta_reward, potential_based_shaping, gated_reward
- Previously tried skeleton family: anchor+proxy+progress+constraint

## Training Feedback (raw evidence)

# Training Feedback

## External evaluation
- score: 188.211666
- episode_length: 405.600000 (mean)
- range: [-134.333448, 272.653537]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.111539 | 0.111539 | 1.000000 | -0.339001 | -0.000010 |
| landing_quality_shaping | 1.965990 | 1.965990 | 0.711869 | 0.000000 | 5.993001 |
| progress_delta_reward | 0.241703 | 0.273100 | 0.997711 | -1.508932 | 1.615135 |
| stability_penalty | -0.129461 | 0.129461 | 1.000000 | -2.551401 | -0.000001 |
| total_reward | 1.966694 | 2.042546 | 1.000000 | -2.541930 | 5.992800 |
| generated_reward | 1.966694 | 2.042546 | 1.000000 | -2.541930 | 5.992800 |
| original_env_reward | -0.178286 | 2.281055 | 1.000000 | -100.000000 | 129.128777 |

## Signals
partial_progress
