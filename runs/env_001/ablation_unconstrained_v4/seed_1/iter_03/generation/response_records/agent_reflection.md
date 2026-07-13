# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- extract current and next state ----
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- distances ----
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # ---- distance progress (strengthened) ----
    progress = prev_dist - next_dist
    c_progress = 5.0 * progress

    # ---- step penalty to discourage lingering ----
    c_step_penalty = -0.05

    # ---- contact reward (without proximity gate) ----
    contact_count = n_left_contact + n_right_contact
    c_contact = 5.0 * contact_count

    # ---- soft landing bonus (one large lump, will only trigger at termination) ----
    landed = (
        n_left_contact > 0.5 and n_right_contact > 0.5 and
        abs(nx) < 0.15 and abs(ny) < 0.15 and
        nvx**2 + nvy**2 < 0.04 and
        abs(n_angle) < 0.2 and
        abs(n_angvel) < 0.2
    )
    c_landing = 200.0 if landed else 0.0

    # ---- speed penalty (global) ----
    speed_sq = nvx**2 + nvy**2
    c_speed = -0.1 * speed_sq

    # ---- posture penalties (global) ----
    c_angle = -0.05 * (n_angle**2)
    c_angvel = -0.005 * (n_angvel**2)

    # ---- total reward ----
    total_reward = (
        c_progress
        + c_step_penalty
        + c_contact
        + c_landing
        + c_speed
        + c_angle
        + c_angvel
    )

    components = {
        'distance_progress': c_progress,
        'step_penalty': c_step_penalty,
        'contact_reward': c_contact,
        'landing_bonus': c_landing,
        'speed_penalty': c_speed,
        'angle_penalty': c_angle,
        'angvel_penalty': c_angvel,
    }

    return (float(total_reward), components)
```
