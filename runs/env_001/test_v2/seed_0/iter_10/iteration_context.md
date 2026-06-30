# Agent Context

- iteration: 10
- target_score: 200.000
- best_score: 188.212 (iter 8)
- current_score: 19.510
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
    "goal_near_oscillation",
    "high_reward_without_success"
  ],
  "hacking_risks": [
    "goal_near_oscillation"
  ],
  "component_analysis": {
    "distance_anchor": {
      "role": "anchor",
      "direction": "negative",
      "signal_strength": "weak",
      "issue": "negative mean but small magnitude, may not be effective"
    },
    "landing_quality_shaping": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "strong",
      "issue": "high mean but external score low, may be rewarding suboptimal behavior"
    },
    "low_speed_bonus": {
      "role": "efficiency",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "none"
    },
    "progress_delta_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "mean positive but external score not improving, may be insufficient"
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "weak",
      "issue": "small magnitude, may not enforce stability"
    }
  },
  "skeleton_assessment": {
    "current_skeleton": [
      "distance_anchor",
      "landing_quality_shaping",
      "low_speed_bonus",
      "progress_delta_reward",
      "stability_penalty"
    ],
    "iterations_on_this_skeleton": 1,
    "best_score_this_skeleton": 19.51,
    "stagnant": false,
    "skeleton_family": "progress+stability+landing_proxy+anchor"
  },
  "recommended_action": "tune",
  "reasoning": "External score 19.51 far below target 200. Landing quality shaping has strong positive signal but external score low, suggesting it may reward near-goal oscillation without success. Progress delta reward positive but insufficient. Recommend tuning coefficients: increase progress delta reward coefficient and reduce landing quality shaping coefficient to prioritize progress over proxy.",
  "new_lessons": [
    "Landing quality shaping can cause goal-near oscillation if not balanced with progress reward.",
    "Progress delta reward coefficient must be high enough to drive meaningful progress."
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
- Previously tried skeleton family: progress+stability+landing_proxy+anchor

## Training Feedback (raw evidence)

# Training Feedback

## External evaluation
- score: 19.509895
- episode_length: 868.300000 (mean)
- range: [-57.088469, 168.481450]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.048918 | 0.048918 | 1.000000 | -0.169501 | -0.000014 |
| landing_quality_shaping | 1.574789 | 1.574789 | 0.753924 | 0.000000 | 3.995997 |
| low_speed_bonus | 0.642045 | 0.642045 | 0.537667 | 0.000000 | 1.998023 |
| progress_delta_reward | 0.309660 | 0.341552 | 0.999046 | -1.969312 | 2.309510 |
| stability_penalty | -0.120864 | 0.120864 | 1.000000 | -2.705785 | -0.000002 |
| total_reward | 2.356712 | 2.398447 | 1.000000 | -3.039735 | 5.992805 |
| generated_reward | 2.356712 | 2.398447 | 1.000000 | -3.039735 | 5.992805 |
| original_env_reward | -0.138805 | 1.881180 | 1.000000 | -100.000000 | 120.737047 |

## Signals
partial_progress
