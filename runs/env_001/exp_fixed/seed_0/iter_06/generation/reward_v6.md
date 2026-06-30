## Revision Decision
- action: mix (tune + add)
- reason: Current reward achieves best_score_so_far (18.32) but is far from target (200). Progress_reward is the dominant signal (0.194 mean) but too weak to drive meaningful improvement. Stability_penalty and distance_anchor are negligible (-0.005 and -0.025). Landing_shaping is not providing enough guidance. Need to significantly strengthen the progress signal while adding a small distance-based reward to prevent drifting.
- required_code_change: 
  1. Increase progress_reward coefficient from 80.0 to 150.0 to amplify the main learning signal
  2. Add a small distance_reward component: -0.1 * next_dist (replacing distance_anchor with slightly stronger pull toward target)
  3. Keep stability_penalty and landing_shaping coefficients unchanged as they are not harmful
- expected_effect: Stronger progress signal should drive faster convergence toward target. Distance_reward provides gentle guidance to stay near target. Stability penalties remain low enough not to dominate.

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
    
    # 1. Main learning signal: progress_delta_reward (strengthened further)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 150.0 * progress_delta  # increased from 80.0
    
    # 2. Reduced stability penalty (unchanged from previous)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.01 * abs(next_body_angle)
    speed_penalty = 0.01 * speed
    ang_vel_penalty = 0.005 * abs(next_ang_vel)
    stability_penalty = -(angle_penalty + speed_penalty + ang_vel_penalty)
    
    # 3. Smooth landing-quality shaping (unchanged from previous)
    near_target = max(0.0, 1.0 - next_dist / 2.0)
    low_speed = max(0.0, 1.0 - speed / 1.0)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.5)
    landing_shaping = 1.0 * near_target + 1.0 * low_speed + 1.0 * stable_angle
    
    # 4. Distance reward to pull agent toward target (slightly stronger than previous distance_anchor)
    distance_reward = -0.1 * next_dist  # increased from -0.05
    
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

## Design Explanation

- **Action**: mix (tune + add)
- **What was retained**: All four component categories from previous reward (progress_reward, stability_penalty, landing_shaping, distance_anchor → distance_reward)
- **What was changed**: 
  - **Tuned**: progress_reward coefficient increased from 80.0 to 150.0 to amplify the main learning signal, since evidence shows it's the dominant nonzero component (0.194 mean) but too weak to drive scores above 18.32
  - **Added**: distance_reward coefficient increased from -0.05 to -0.1 (renamed from distance_anchor to distance_reward for clarity) to provide stronger pull toward target
- **Why this change**: The best score (18.32) is far from target (200). Progress_reward is the only component with meaningful magnitude (0.194 mean), but it's insufficient. Strengthening it by ~1.9x should help the agent learn faster. The distance_reward provides gentle guidance to stay near target without dominating.
- **Why no terminal rewards**: Still no explicit success/failure signals available in info
- **Next observation focus**: Monitor progress_reward mean and nonzero rate; if it increases significantly but score doesn't improve, consider adding a landing completion bonus or adjusting landing_shaping coefficients