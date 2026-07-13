# Response Record

# 设计理由
velocity_penalty 在当前训练中 `mean=0.0`，说明从未触发，无法提供靠近地面时减速的梯度。修改为降低速度阈值至 2.0、减小地面邻近衰减系数（scale 3.0），并增加 0.2 的系数，使惩罚在接近地面时对超标速度连续生效，每步惩罚量控制在主进展信号（per‑step ≈ 0.26）的 0.1~0.3 倍，不会过度压制下降。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Gym-style reward designed for lunar-lander-like 2D vehicle.
    Focus: proximity + gated stability + gentle contact proxy.
    """
    # --- Unpack states ---
    x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, left_contact, right_contact = obs
    nx_pos, ny_pos, nx_vel, ny_vel, nbody_angle, nang_vel, nleft_contact, nright_contact = next_obs

    # Constants
    current_sq_dist = x_pos ** 2 + y_pos ** 2
    next_sq_dist = nx_pos ** 2 + ny_pos ** 2

    PROXIMITY_SCALE = 10.0
    ANGLE_GATE_K = 15.0
    # ---------- velocity penalty parameters (scaled for earlier activation) ----------
    VEL_THRESH = 2.0
    VEL_PENALTY_DIST_SCALE = 3.0
    VEL_PENALTY_COEFF = 0.2
    # --------------------------------------------------------------------------------
    CONTACT_PROXY_K = 4.0
    FUEL_COST_WEIGHT = 0.15

    # 1. Goal Proximity (mandatory)
    dist_improvement = current_sq_dist - next_sq_dist
    proximity_reward = PROXIMITY_SCALE * dist_improvement / (1.0 + abs(dist_improvement))

    # 2. Orientation Stability Gate
    angle_error = abs(nbody_angle)
    angle_gate = 2.718281828 ** (-ANGLE_GATE_K * angle_error ** 2)
    gated_proximity = proximity_reward * angle_gate

    # 3. Soft Landing Velocity Penalty (now active from higher altitude)
    speed = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    excess_speed = max(0.0, speed - VEL_THRESH)
    ground_proximity = 1.0 / (1.0 + VEL_PENALTY_DIST_SCALE * abs(ny_pos))
    velocity_penalty = -VEL_PENALTY_COEFF * excess_speed * ground_proximity

    # 4. Soft Landing Proxy
    leg_contact_both = min(nleft_contact, nright_contact)
    slow_factor = 1.0 / (1.0 + CONTACT_PROXY_K * speed)
    near_ground_factor = 1.0 / (1.0 + 4.0 * abs(ny_pos))
    contact_proxy = (leg_contact_both * slow_factor * near_ground_factor) ** (1.0 / 3.0)
    soft_landing_bonus = 2.0 * contact_proxy

    # 5. Action Efficiency
    is_engine_on = 0.0 if action == 0 else 1.0
    fuel_cost = -FUEL_COST_WEIGHT * is_engine_on

    total_reward = gated_proximity + velocity_penalty + soft_landing_bonus + fuel_cost

    components = {
        "proximity_reward": gated_proximity,
        "velocity_penalty": velocity_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "fuel_cost": fuel_cost
    }

    return float(total_reward), components
```

# 诊断摘要
- **evidence**: velocity_penalty mean = 0.0 (never triggered), score = -110.82 despite +18 proximity reward, terminated=20/20 implies frequent crash/out-of-bounds.
- **behavior**: agent achieves sizable distance improvement but fails to slow down near ground, leading to crash termination and highly negative outcomes.
- **signal**: no effective downward velocity regulation in the approach phase; the existing penalty is too late (high threshold) and too local (steep distance gate).
- **level**: Level 1 — adjust threshold/scale/coefficient of the existing velocity penalty component.
- **hypothesis**: lowering the speed threshold and extending the ground‑proximity influence will give the agent continuous, gentle pressure to decelerate before landing, reducing crash failures and improving score.
- **risk**: overly cautious early deceleration may lengthen episode slightly or cause hovering, but the mild coefficient keeps per‑step penalty within a safe fraction of progress reward.
