# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]

    distance = (x**2 + y**2)**0.5
    proximity_reward = 1.0 / (1.0 + distance)

    tilt_penalty = -0.5 * abs(angle)
    speed = (vx**2 + vy**2)**0.5
    velocity_penalty = -0.1 * speed
    rotation_penalty = -0.1 * abs(angular_vel)

    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = proximity_reward + tilt_penalty + velocity_penalty + rotation_penalty + fuel_penalty

    components = {
        "proximity_reward": proximity_reward,
        "tilt_penalty": tilt_penalty,
        "velocity_penalty": velocity_penalty,
        "rotation_penalty": rotation_penalty,
        "fuel_penalty": fuel_penalty
    }
    return total_reward, components
```
