```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Potential-based shaping reward for the 2D lander task.
    The potential penalises distance, speed, tilt and angular velocity,
    and rewards leg‑contact, all in a single continuous function.
    The reward is the improvement in potential from obs to next_obs.
    """
    # Weights – can be tuned later
    w_dist  = 5.0
    w_vel   = 1.0
    w_angle = 2.0
    w_angvel= 1.0
    w_contact = 2.0

    def potential(o):
        x = o[0]
        y = o[1]
        vx = o[2]
        vy = o[3]
        angle = o[4]
        ang_vel = o[5]
        left_c  = o[6]
        right_c = o[7]

        distance = (x**2 + y**2)**0.5
        speed    = (vx**2 + vy**2)**0.5
        contact  = left_c + right_c   # 0, 1 or 2

        # Combine all objectives into a single scalar (higher is better)
        return -(w_dist * distance
                 + w_vel * speed
                 + w_angle * abs(angle)
                 + w_angvel * abs(ang_vel)
                 - w_contact * contact)

    pot_current = potential(obs)
    pot_next    = potential(next_obs)

    reward_delta = pot_next - pot_current

    # For numerical stability we can clamp the delta, but it’s usually small
    reward = reward_delta

    reward_components = {
        "progress_delta": reward
    }

    return float(reward), reward_components
```