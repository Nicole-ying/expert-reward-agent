# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack states
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Encourage downward motion towards the platform (y decreases toward 0)
    descent_reward = 0.8 * (y_prev - y)

    # Penalize horizontal deviation from the target center
    horiz_penalty = -0.2 * abs(x)

    # Penalize high horizontal and vertical speeds to encourage a soft landing
    vel_penalty = -0.1 * (abs(vx) + abs(vy))

    # Penalize deviation from upright orientation
    orient_penalty = -0.1 * abs(body_angle)

    # Penalize angular velocity to suppress spinning
    angvel_penalty = -0.05 * abs(angvel)

    # Reward each foot touching the ground to promote contact
    contact_reward = (left_contact + right_contact) * 2.0

    # Sparse large bonus for a successful soft landing
    contact_ok = float(left_contact > 0.5 and right_contact > 0.5)
    speed_ok = float(abs(vx) < 0.2 and abs(vy) < 0.2 and abs(angvel) < 0.1)
    angle_ok = float(abs(body_angle) < 0.1)
    landing_bonus = 200.0 * contact_ok * speed_ok * angle_ok

    # Small per‑step penalty to discourage lingering
    time_penalty = -0.01

    total = (descent_reward + horiz_penalty + vel_penalty +
             orient_penalty + angvel_penalty + contact_reward +
             landing_bonus + time_penalty)

    components = {
        "descent_reward": descent_reward,
        "horiz_penalty": horiz_penalty,
        "vel_penalty": vel_penalty,
        "orient_penalty": orient_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "time_penalty": time_penalty
    }
    return total, components
```
