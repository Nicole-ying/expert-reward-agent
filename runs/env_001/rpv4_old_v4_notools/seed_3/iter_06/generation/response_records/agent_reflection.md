# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos, y_pos = obs[0], obs[1]
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_body_angle, n_ang_vel = next_obs[4], next_obs[5]
    n_left_contact, n_right_contact = next_obs[6], next_obs[7]

    # Distance to target platform (center at 0,0)
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    n_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # ---------- 1. Main progress signal: improvement delta ----------
    progress_reward = 2.0 * (dist - n_dist)  # positive when approaching

    # ---------- 2. Soft landing proxy: joint condition average ----------
    # Proximity factor
    prox = 1.0 / (1.0 + 5.0 * n_dist)
    # Contact factor
    contact = (n_left_contact + n_right_contact) / 2.0
    # Angle factor
    angle_factor = max(0.0, 1.0 - abs(n_body_angle) / 0.8)
    # Angular velocity factor
    angvel_factor = max(0.0, 1.0 - abs(n_ang_vel) / 2.0)
    # Speed factor
    speed = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    speed_factor = max(0.0, 1.0 - speed / 2.0)

    landing_proxy = (prox + contact + angle_factor + angvel_factor + speed_factor) / 5.0
    landing_reward = 10.0 * landing_proxy

    # ---------- 3. Crash / boundary constraint: hinge penalties ----------
    # Horizontal drift penalty
    x_violation = max(0.0, abs(nx_pos) - 1.5)
    # Excessive downward velocity penalty
    v_violation = max(0.0, -ny_vel - 3.0)
    crash_penalty = -1.0 * x_violation - 1.0 * v_violation

    total_reward = progress_reward + landing_reward + crash_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
        "crash_penalty": crash_penalty,
    }
    return float(total_reward), components
```
