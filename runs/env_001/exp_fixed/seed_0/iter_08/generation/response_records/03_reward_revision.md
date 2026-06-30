# Response Record

## Revision Decision
- action: tune
- reason: Previous reward (iter_7) collapsed to -15.09 from best 146.23. Component evidence shows progress_reward is still active (0.537 mean, 99.9% nonzero) but the distance_reward was strengthened too aggressively (-0.5 * next_dist), likely dominating and pulling the agent into bad states. The landing_shaping coefficients are too weak to counteract this. Need to reduce distance_reward back to a small anchor, increase landing_shaping to provide better approach guidance, and slightly increase progress_reward coefficient to strengthen the main learning signal.
- required_code_change: Reduce distance_reward coefficient from -0.5 to -0.1; increase landing_shaping coefficients from 1.0 to 2.0 for near_target and low_speed, 1.5 for stable_angle; increase progress_reward coefficient from 150.0 to 200.0
- expected_effect: Restore the balance that worked in iter_6 (best 146.23) while adding slightly stronger landing guidance to prevent the collapse. The progress signal should dominate again, with distance_reward as a small anchor only.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. Main learning signal: progress_delta_reward (strengthened to restore dominance)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 200.0 * progress_delta
    
    # 2. Reduced stability penalty (unchanged - minimal impact)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.01 * abs(next_body_angle)
    speed_penalty = 0.01 * speed
    ang_vel_penalty = 0.005 * abs(next_ang_vel)
    stability_penalty = -(angle_penalty + speed_penalty + ang_vel_penalty)
    
    # 3. Strengthened landing-quality shaping (increased coefficients for better approach guidance)
    near_target = max(0.0, 1.0 - next_dist / 2.0)
    low_speed = max(0.0, 1.0 - speed / 1.0)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    landing_shaping = 2.0 * near_target + 2.0 * low_speed + 1.5 * stable_angle
    
    # 4. Distance reward - reduced back to small anchor (was -0.5, now -0.1)
    distance_reward = -0.1 * next_dist
    
    # Combine components
    total_reward = progress_reward + stability_penalty + landing_shaping + distance_reward
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "distance_reward": distance_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

## Design Notes

- **Action**: tune — adjusted coefficients only, kept all four components from previous reward.
- **Retained**: progress_reward (strengthened to 200.0), stability_penalty (unchanged minimal), landing_shaping (kept but coefficients increased), distance_reward (kept as small anchor).
- **Changed**: Reduced distance_reward from -0.5 to -0.1 (was too aggressive in iter_7, likely pulling agent into bad states). Increased landing_shaping coefficients from 1.0 to 2.0 for near_target and low_speed, 1.5 for stable_angle (to provide stronger approach guidance). Increased progress_reward from 150.0 to 200.0 (to restore dominance of the main learning signal).
- **Why this targets the failure**: The collapse from 146.23 to -15.09 coincided with strengthening distance_reward to -0.5. This likely created a strong pull toward target that overwhelmed the progress signal, causing the agent to rush toward target without learning proper approach. Reducing distance_reward back to -0.1 and strengthening landing_shaping should restore the balance that worked in iter_6 while adding better approach guidance.
- **Why no terminal rewards**: Still no explicit success/failure signals available in info.
- **Next observation focus**: Monitor if progress_reward mean stays above 0.5 and if landing_shaping mean increases from near-zero. If score recovers toward 146, this tuning is on the right track.
