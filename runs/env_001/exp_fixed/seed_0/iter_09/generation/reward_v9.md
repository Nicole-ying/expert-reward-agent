## Revision Decision
- action: tune
- reason: Previous reward collapsed from best 146.23 to -18.55. The progress_reward coefficient (200.0) is too aggressive, causing the agent to prioritize speed over stability and landing quality. The landing_shaping coefficients are too weak to compensate. Need to rebalance: reduce progress dominance, strengthen landing guidance, and add a small distance anchor as suggested by skeleton plan.
- required_code_change: Reduce progress_reward coefficient from 200.0 to 80.0; increase landing_shaping coefficients (near_target from 2.0 to 4.0, low_speed from 2.0 to 3.0, stable_angle from 1.5 to 2.0); increase distance_reward from -0.1 to -0.5 to provide stronger proximity incentive; keep stability_penalty minimal.
- expected_effect: Less aggressive progress chasing should prevent crashes; stronger landing shaping should guide smooth approach; distance anchor should encourage getting closer to target.

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
    
    # 1. Progress delta reward (reduced coefficient to prevent aggressive speed)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 80.0 * progress_delta
    
    # 2. Minimal stability penalty (unchanged - negligible impact)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.01 * abs(next_body_angle)
    speed_penalty = 0.01 * speed
    ang_vel_penalty = 0.005 * abs(next_ang_vel)
    stability_penalty = -(angle_penalty + speed_penalty + ang_vel_penalty)
    
    # 3. Strengthened landing-quality shaping (increased coefficients for better approach guidance)
    near_target = max(0.0, 1.0 - next_dist / 2.0)
    low_speed = max(0.0, 1.0 - speed / 1.0)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    landing_shaping = 4.0 * near_target + 3.0 * low_speed + 2.0 * stable_angle
    
    # 4. Distance reward (increased to provide stronger proximity incentive)
    distance_reward = -0.5 * next_dist
    
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