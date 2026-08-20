```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Potential‑based shaping hyper‑parameters
    w_vel = 0.1            # weight for absolute linear velocity sum
    w_angle = 0.5          # weight for absolute body angle
    shaping_scale = 50.0   # scales the shaping signal
    gamma = 0.99           # discount factor for shaping

    # Landing continuous reward hyper‑parameters
    alpha = 5.0            # x‑position penalty (0 = target)
    beta = 5.0             # y‑position penalty
    delta = 1.0            # speed penalty exponent coefficient
    eta = 10.0             # angle penalty exponent coefficient

    # ---------- potential functions ----------
    dist_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    vel_curr = abs(obs[2]) + abs(obs[3])
    vel_next = abs(next_obs[2]) + abs(next_obs[3])

    angle_curr = abs(obs[4])
    angle_next = abs(next_obs[4])

    phi_curr = -(dist_curr + w_vel * vel_curr + w_angle * angle_curr)
    phi_next = -(dist_next + w_vel * vel_next + w_angle * angle_next)

    shaping_reward = shaping_scale * (gamma * phi_next - phi_curr)

    # ---------- landing continuous reward ----------
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    landing_reward = 0.0

    if left_contact > 0.5 and right_contact > 0.5:
        x = next_obs[0]
        y = next_obs[1]
        vx = next_obs[2]
        vy = next_obs[3]
        angle = next_obs[4]

        score = (2.718281828 ** (-alpha * x ** 2)) * \
                (2.718281828 ** (-beta * y ** 2)) * \
                (2.718281828 ** (-delta * (vx ** 2 + vy ** 2))) * \
                (2.718281828 ** (-eta * angle ** 2))
        landing_reward = score

    total_reward = shaping_reward + landing_reward

    components = {
        'shaping_reward': shaping_reward,
        'landing_reward': landing_reward
    }

    return float(total_reward), components
```