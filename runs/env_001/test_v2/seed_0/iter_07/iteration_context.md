# Agent Context

- iteration: 7
- target_score: 200.000
- best_score: 93.790 (iter 6)
- current_score: 93.790
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
      "issue": "negative mean indicates agent is far from target; may be too weak to pull agent closer"
    },
    "landing_quality_shaping": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "low nonzero rate (0.32) suggests sparse activation; may not guide consistently"
    },
    "progress_delta_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "positive mean but small magnitude; may be insufficient to drive progress"
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "moderate",
      "issue": "negative mean but small magnitude; may be too weak to enforce stability"
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
    "best_score_this_skeleton": 93.79,
    "stagnant": false,
    "skeleton_family": "anchor+proxy+progress+constraint"
  },
  "recommended_action": "tune",
  "reasoning": "External score 93.79 is far below target 200. Progress_delta_reward mean 0.126 is low; distance_anchor negative (-0.082) indicates agent not reaching target. Landing_quality_shaping nonzero rate 0.32 suggests sparse guidance. No stagnation yet (only 1 iteration). Recommend tuning coefficients: increase progress_delta_reward coefficient, increase distance_anchor magnitude, and increase landing_quality_shaping activation frequency.",
  "new_lessons": [
    "progress_delta_reward coefficient may need to be increased to drive stronger progress",
    "distance_anchor negative mean suggests agent is not being pulled toward target; consider increasing its weight"
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
- score: 93.789748
- episode_length: 433.800000 (mean)
- range: [-63.747210, 247.942875]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.082026 | 0.082026 | 1.000000 | -0.169501 | -0.000009 |
| landing_quality_shaping | 0.307413 | 0.307413 | 0.318349 | 0.000000 | 2.994347 |
| progress_delta_reward | 0.126131 | 0.135551 | 0.999793 | -0.377514 | 0.406659 |
| stability_penalty | -0.102975 | 0.102975 | 1.000000 | -1.296748 | -0.000000 |
| total_reward | 0.248544 | 0.353085 | 1.000000 | -1.313760 | 2.994109 |
| generated_reward | 0.248544 | 0.353085 | 1.000000 | -1.313760 | 2.994109 |
| original_env_reward | -0.671153 | 2.555039 | 1.000000 | -100.000000 | 125.946996 |

## Signals
partial_progress
