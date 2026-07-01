# Analysis Report

## Recommended Action: tune
Current score 67.07 is best so far, but far from target 200. Landing_shaping dominates (mean 2.049) while progress_reward is weak (mean 0.206). This suggests the agent is exploiting relaxed landing conditions (contact reward hacking) without making real progress toward the goal. The skeleton is not stagnant (only 2 iterations), so tuning is appropriate: reduce landing_shaping coefficient and tighten conditions to prevent hacking, and increase progress_reward coefficient further to drive learning.

## Skeleton Status
- family: progress+stability+landing_proxy
- stagnant: False
- iterations_on_skeleton: 2

## Component Analysis
- progress_reward: role=progress dir=positive issue=mean 0.206 is low relative to coefficient 50.0; agent may be oscillating near goal, causing small net progress
- stability_penalty: role=constraint dir=negative issue=mean -0.079 is within acceptable range, not dominating
- landing_shaping: role=proxy dir=positive issue=mean 2.049 dominates total reward; conditions may be too relaxed, leading to contact reward hacking without true landing

## Detected Issues
- failure_modes: goal_near_oscillation, high_reward_without_success
- hacking_risks: contact_reward_hacking
