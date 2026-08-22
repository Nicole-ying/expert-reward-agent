```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs and next_obs fields: 0-x, 1-y, 2-vx, 3-vy, 4-angle, 5-angvel, 6-left_contact, 7-right_contact
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Shaping: encourage moving closer to the target (origin)
    prev_dist = (x_prev**2 + y_prev**2) ** 0.5
    dist = (x**2 + y**2) ** 0.5
    shaping_reward = 0.05 * (prev_dist - dist)

    # Reward each touching foot to promote ground contact
    contact_reward = (left_contact + right_contact) * 2.0

    # Large sparse bonus for a successful soft landing
    contact_ok = float(left_contact > 0.5 and right_contact > 0.5)
    speed_ok = float(abs(vx) < 0.2 and abs(vy) < 0.2 and abs(angvel) < 0.1)
    angle_ok = float(abs(body_angle) < 0.1)
    landing_bonus = 200.0 * contact_ok * speed_ok * angle_ok

    # Small per‑step penalty to discourage lingering
    time_penalty = -0.01

    total = shaping_reward + contact_reward + landing_bonus + time_penalty
    components = {
        "shaping_reward": shaping_reward,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "time_penalty": time_penalty
    }
    return total, components
```