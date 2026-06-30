## Revision Decision
- action: tune
- reason: Previous reward shows progress_reward is active (nonzero_rate ~1.0) but the overall score is still negative (-41.34) and below best (-36.73). The progress signal is too weak to drive meaningful improvement, while stability_penalty and distance_anchor are too strong relative to progress. Need to significantly increase progress_reward coefficient and reduce penalties to let progress dominate.
- required_code_change: 
  1. Increase progress_reward coefficient from 30.0 to 80.0
  2. Reduce stability_penalty coefficients: angle_penalty from 0.05 to 0.01, speed_penalty from 0.04 to 0.01, ang_vel_penalty from 0.02 to 0.005
  3. Reduce landing_shaping coefficients from 2.0 to 1.0 each
  4. Reduce distance_anchor from -0.2 to -0.05
- expected_effect: Progress signal will dominate (~80x delta vs ~0.02 penalties), allowing the agent to learn to move toward target. The penalties are still present to prevent extreme behaviors but won't overwhelm the main learning signal.

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
    
    # 1. Main learning signal: progress_delta_reward (significantly strengthened)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 80.0 * progress_delta  # increased from 30.0
    
    # 2. Reduced stability penalty (further weakened to avoid dominating)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.01 * abs(next_body_angle)       # reduced from 0.05
    speed_penalty = 0.01 * speed                      # reduced from 0.04
    ang_vel_penalty = 0.005 * abs(next_ang_vel)       # reduced from 0.02
    stability_penalty = -(angle_penalty + speed_penalty + ang_vel_penalty)
    
    # 3. Smooth landing-quality shaping (reduced coefficients to avoid dominating)
    near_target = max(0.0, 1.0 - next_dist / 2.0)     # 1.0 when dist=0, 0.0 when dist>=2
    low_speed = max(0.0, 1.0 - speed / 1.0)           # 1.0 when speed=0, 0.0 when speed>=1
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.5)  # 1.0 when angle=0, 0.0 when angle>=0.5
    landing_shaping = 1.0 * near_target + 1.0 * low_speed + 1.0 * stable_angle  # reduced from 2.0
    
    # 4. Distance anchor to prevent drifting far away (reduced)
    distance_anchor = -0.05 * next_dist  # reduced from -0.2
    
    # Combine components
    total_reward = progress_reward + stability_penalty + landing_shaping + distance_anchor
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

## Design Notes

- **Action**: tune — adjusted coefficients only, no structural changes.
- **Retained**: All four components from previous reward (progress_reward, stability_penalty, landing_shaping, distance_anchor).
- **Weakened**: stability_penalty coefficients reduced by ~75%, landing_shaping reduced by 50%, distance_anchor reduced by 75%.
- **Strengthened**: progress_reward coefficient increased from 30.0 to 80.0 (2.67x).
- **Why this change**: The evidence shows progress_reward has a very high nonzero_rate (0.999981) but its mean is only 0.067, meaning the agent is making tiny progress steps. The penalties (-0.022 stability, -0.114 distance) are collectively larger than the progress signal. By making progress ~80x stronger than penalties, the agent should learn to prioritize moving toward the target.
- **Why no terminal rewards**: Still no explicit success/failure signals available in info.
- **Next observation**: Monitor if progress_reward mean increases significantly (target >1.0) and if external evaluation score becomes positive. If progress_reward mean stays low, consider adding a distance-based shaping bonus.