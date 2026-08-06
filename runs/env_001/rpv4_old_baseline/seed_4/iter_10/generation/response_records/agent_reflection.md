# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Derived quantities
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    speed_next = (vx_next ** 2 + vy_next ** 2) ** 0.5
    height_curr = y_curr + 1e-6  # avoid division by zero

    # 1. Approach shaping toward the target
    approach_reward = 5.0 * (dist_curr - dist_next)

    # 2. Height-dependent speed penalty: high speed allowed at height, strong penalty near ground
    speed_penalty = -0.1 * (speed_next ** 2) / (height_curr + 0.2)

    # 3. Attitude and angular velocity penalties
    angle_penalty = -0.5 * abs(angle_next)
    angvel_penalty = -0.3 * (angvel_next ** 2)

    # 4. Centering penalty
    centering_penalty = -0.1 * abs(x_next)

    # 5. Reduced engine penalty to encourage control
    if action == 2:          # main engine
        engine_penalty = -0.05
    elif action in (1, 3):   # side engines
        engine_penalty = -0.02
    else:
        engine_penalty = 0.0

    # 6. Time penalty (instead of survival bonus) to encourage faster completion
    time_penalty = -0.05

    # 7. Soft landing reward – triggered on both legs contact, strongly shaped by speed and angle
    both_contact = float(left_contact * right_contact)
    landing_base = 250.0
    landing_speed_penalty = -300.0 * speed_next
    landing_angle_penalty = -200.0 * abs(angle_next)
    landing_reward = both_contact * (landing_base + landing_speed_penalty + landing_angle_penalty)

    total_reward = (approach_reward +
                    speed_penalty +
                    angle_penalty +
                    angvel_penalty +
                    centering_penalty +
                    engine_penalty +
                    time_penalty +
                    landing_reward)

    components = {
        "approach_reward": approach_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "centering_penalty": centering_penalty,
        "engine_penalty": engine_penalty,
        "time_penalty": time_penalty,
        "landing_reward": landing_reward
    }
    return float(total_reward), components
```
