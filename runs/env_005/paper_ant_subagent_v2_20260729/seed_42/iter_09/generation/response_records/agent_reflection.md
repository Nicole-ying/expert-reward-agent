# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract key signals from next_obs
    body_z = next_obs[0]
    quat_w = next_obs[1]   # not used directly but kept for completeness
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]   # not used directly
    
    forward_vel = next_obs[13]
    lateral_vel = next_obs[14]
    vertical_vel = next_obs[15]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    
    # Derived upright metric: body_up_z
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    
    # --- Reward Components ---
    
    # Main progress: forward velocity (dense, linear)
    forward_reward = 5.0 * forward_vel
    
    # Predictive descent penalty: downward velocity when height is losing margin
    descent_penalty = 0.0
    if body_z < 0.5 and vertical_vel < 0.0:
        descent_penalty = -0.5 * abs(vertical_vel) * (0.5 - body_z)
    
    # Predictive ascent penalty: upward velocity when height is too high
    ascent_penalty = 0.0
    if body_z > 0.8 and vertical_vel > 0.0:
        ascent_penalty = -0.3 * vertical_vel * (body_z - 0.8)
    
    # Lateral drift hinge: penalise excessive sideways motion
    lateral_penalty = -0.2 * max(0.0, abs(lateral_vel) - 0.3)
    
    # Roll / pitch angular velocity hinge (avoids rapid tilting)
    roll_vel_penalty = -0.1 * max(0.0, abs(roll_vel) - 3.0)
    pitch_vel_penalty = -0.1 * max(0.0, abs(pitch_vel) - 3.0)
    
    # Upright tilt penalty: punish large body tilt
    upright_penalty = -0.5 * max(0.0, 0.6 - body_up_z)
    
    # Joint velocity regularisation (very light) — expanded to avoid slices
    joint_vel_penalty = -0.0005 * (
        next_obs[19]**2 + next_obs[20]**2 + next_obs[21]**2 + next_obs[22]**2 +
        next_obs[23]**2 + next_obs[24]**2 + next_obs[25]**2 + next_obs[26]**2
    )
    
    # Action magnitude regularisation (tiny, analogous to energy cost)
    action_penalty = -0.001 * sum(a**2 for a in action)
    
    total = (forward_reward + descent_penalty + ascent_penalty +
             lateral_penalty + roll_vel_penalty + pitch_vel_penalty +
             upright_penalty + joint_vel_penalty + action_penalty)
    
    components = {
        'forward_reward': forward_reward,
        'descent_penalty': descent_penalty,
        'ascent_penalty': ascent_penalty,
        'lateral_penalty': lateral_penalty,
        'roll_vel_penalty': roll_vel_penalty,
        'pitch_vel_penalty': pitch_vel_penalty,
        'upright_penalty': upright_penalty,
        'joint_vel_penalty': joint_vel_penalty,
        'action_penalty': action_penalty
    }
    
    return float(total), components
```
