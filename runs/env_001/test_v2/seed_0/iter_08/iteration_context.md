# Agent Context

- iteration: 8
- target_score: 200.000
- best_score: 110.984 (iter 7)
- current_score: 110.984
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
    "goal_near_oscillation"
  ],
  "component_analysis": {
    "distance_anchor": {
      "role": "anchor",
      "direction": "negative",
      "signal_strength": "strong",
      "issue": "always negative, may discourage exploration"
    },
    "landing_quality_shaping": {
      "role": "proxy",
      "direction": "positive",
      "signal_strength": "strong",
      "issue": "none"
    },
    "progress_delta_reward": {
      "role": "progress",
      "direction": "positive",
      "signal_strength": "moderate",
      "issue": "low mean, may need higher coefficient"
    },
    "stability_penalty": {
      "role": "constraint",
      "direction": "negative",
      "signal_strength": "moderate",
      "issue": "none"
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
    "best_score_this_skeleton": 110.98,
    "stagnant": false,
    "skeleton_family": "anchor+proxy+progress+constraint"
  },
  "recommended_action": "tune",
  "reasoning": "当前骨架仅运行1轮，得分110.98，未达目标200。progress_delta_reward均值低（0.147），建议增大其系数；distance_anchor始终为负，可能抑制探索，可降低其权重。landing_quality_shaping和stability_penalty表现正常。",
  "new_lessons": [
    "progress_delta_reward coefficient should be increased to drive more progress",
    "distance_anchor negative bias may hinder exploration; consider reducing its weight"
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
- score: 110.983632
- episode_length: 495.800000 (mean)
- range: [-45.099904, 265.964283]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.299615 | 0.299615 | 1.000000 | -0.847504 | -0.000019 |
| landing_quality_shaping | 1.313273 | 1.313273 | 0.667698 | 0.000000 | 4.994297 |
| progress_delta_reward | 0.146532 | 0.163949 | 0.999920 | -0.684212 | 0.794032 |
| stability_penalty | -0.152022 | 0.152022 | 1.000000 | -2.524427 | -0.000004 |
| total_reward | 1.008169 | 1.390079 | 1.000000 | -2.535247 | 4.993656 |
| generated_reward | 1.008169 | 1.390079 | 1.000000 | -2.535247 | 4.993656 |
| original_env_reward | -0.327450 | 3.209037 | 1.000000 | -100.000000 | 134.926482 |

## Signals
partial_progress
