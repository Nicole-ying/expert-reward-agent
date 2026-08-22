# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract observation indices ----------
    body_z_next = next_obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_vel = obs[13]
    body_y_vel = obs[14]

    # ---------- forward velocity reward (primary) ----------
    w_fwd = 1.5
    forward_reward = w_fwd * body_x_vel

    # ---------- height safety gate (dual-bound) ----------
    z_low_safe = 0.35
    z_low_dead = 0.2
    z_high_safe = 0.85
    z_high_dead = 1.0

    gate_low = min(1.0, max(0.0, (body_z_next - z_low_dead) / (z_low_safe - z_low_dead)))
    gate_high = min(1.0, max(0.0, (z_high_dead - body_z_next) / (z_high_dead - z_high_safe)))
    height_gate = gate_low * gate_high

    # ---------- gated forward ----------
    gated_forward = forward_reward * height_gate

    # ---------- upright posture reward (light) ----------
    body_up = 1.0 - 2.0 * (qx * qx + qy * qy)
    w_up = 0.1
    upright_reward = w_up * body_up

    # ---------- lateral drift penalty (light) ----------
    w_lat = 0.3
    lateral_penalty = -w_lat * (body_y_vel ** 2)

    # ---------- joint velocity smoothness penalty (new, light) ----------
    # individual joint velocity indices (19..26), no slicing
    jv0 = obs[19]
    jv1 = obs[20]
    jv2 = obs[21]
    jv3 = obs[22]
    jv4 = obs[23]
    jv5 = obs[24]
    jv6 = obs[25]
    jv7 = obs[26]
    joint_vel_sq_sum = jv0*jv0 + jv1*jv1 + jv2*jv2 + jv3*jv3 + jv4*jv4 + jv5*jv5 + jv6*jv6 + jv7*jv7
    w_joint_vel = 0.01
    joint_vel_penalty = -w_joint_vel * joint_vel_sq_sum

    # ---------- total reward ----------
    total_reward = gated_forward + upright_reward + lateral_penalty + joint_vel_penalty

    components = {
        "gated_forward": gated_forward,
        "height_gate": height_gate,
        "upright_reward": upright_reward,
        "lateral_penalty": lateral_penalty,
        "joint_vel_penalty": joint_vel_penalty
    }
    return float(total_reward), components
```
