# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next state
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

    # Strong potential-based shaping: reward for reducing distance to target
    approach_reward = 20.0 * (dist_curr - dist_next)

    # Per-step contact reward: only when both legs touch the platform
    both_legs = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    # Landing quality: horizontal speed, vertical speed, and body angle all close to zero
    contact_quality = max(0.0, 1.0 - (abs(vx_next) + abs(vy_next) + abs(angle_next)))
    contact_reward = 50.0 * both_legs * contact_quality

    # Moderate state penalties (guide toward slow, upright, non-spinning descent)
    speed_penalty = -0.5 * speed_next
    angle_penalty = -1.0 * (angle_next ** 2)
    angvel_penalty = -0.5 * (angvel_next ** 2)

    # Small time penalty to encourage fast landing
    survival_penalty = -0.1

    # Fuel penalties
    main_penalty = -0.3 if action == 2 else 0.0
    side_penalty = -0.03 if action in (1, 3) else 0.0
    engine_penalty = main_penalty + side_penalty

    total_reward = (approach_reward +
                    contact_reward +
                    speed_penalty +
                    angle_penalty +
                    angvel_penalty +
                    survival_penalty +
                    engine_penalty)

    components = {
        "approach_reward": approach_reward,
        "contact_reward": contact_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "survival_penalty": survival_penalty,
        "engine_penalty": engine_penalty
    }
    return float(total_reward), components
```
