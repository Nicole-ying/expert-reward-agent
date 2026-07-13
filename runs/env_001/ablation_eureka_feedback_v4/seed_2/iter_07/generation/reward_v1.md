# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Gym-style reward designed for lunar-lander-like 2D vehicle.
    Focus: proximity + gated stability + gentle contact proxy.
    """
    # --- Unpack states ---
    # Current (pre-step)
    x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, left_contact, right_contact = obs

    # Next (post-step)
    nx_pos, ny_pos, nx_vel, ny_vel, nbody_angle, nang_vel, nleft_contact, nright_contact = next_obs

    # Constants
    # Square distance as convexification of linear progress (breaks local plateaus)
    current_sq_dist = x_pos ** 2 + y_pos ** 2
    next_sq_dist = nx_pos ** 2 + ny_pos ** 2

    # Gating thresholds (soft, designed for safety around landing zone)
    # 5.0 is generous, allows distant approach while heavily rewarding proximity
    PROXIMITY_SCALE = 10.0
    # 0.5 radians is ~28 degrees, beyond which gate starts to choke progress
    ANGLE_GATE_K = 15.0
    # 5.0 m/s is safe touch; hinge penalizes velocity above this linearly
    VEL_THRESH = 5.0
    # Ground proximity activation: velocity penalty scales with 1/y_pos when close
    VEL_PENALTY_DIST_SCALE = 8.0
    # Contact proxy smoothing
    CONTACT_PROXY_K = 4.0
    # Fuel/action cost (mild)
    FUEL_COST_WEIGHT = 0.15

    # 1. Goal Proximity (mandatory) — convexified distance progress
    # Uses improvement_delta on squared distance: creates strong gradient near origin
    dist_improvement = current_sq_dist - next_sq_dist  # positive means got closer
    # Compress improvement to avoid spam rewards when already at goal
    proximity_reward = PROXIMITY_SCALE * dist_improvement / (1.0 + abs(dist_improvement))

    # 2. Orientation Stability Gate (soft health gate for primary progress)
    # The idea: posture error attenuates proximity_reward. If agent is tilted, it can't
    # collect full progress — this forces posture correction *before* arrival, not after.
    angle_error = abs(nbody_angle)  # 0 is ideal horizontal
    # Exponential gate: near 1.0 when angle small, falls rapidly beyond 0.05 rad
    angle_gate = 2.718281828 ** (-ANGLE_GATE_K * angle_error ** 2)
    gated_proximity = proximity_reward * angle_gate

    # 3. Soft Landing Velocity Penalty (conditional on proximity to ground)
    # Only meaningful when near surface (small ny_pos). Avoid penalizing fast descent
    # from high altitude.
    # We use a soft hinge on speed magnitude, weighted by ground proximity.
    speed = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    # Hinge: only penalizes speed above VEL_THRESH, linearly
    excess_speed = max(0.0, speed - VEL_THRESH)
    # Ground proximity weight: 1/(1 + scale * ny_pos) → peaks when close to ground
    ground_proximity = 1.0 / (1.0 + VEL_PENALTY_DIST_SCALE * abs(ny_pos))
    velocity_penalty = -excess_speed * ground_proximity

    # 4. Soft Landing Proxy (joint condition with continuous factors)
    # Encourages simultaneous: both legs + low speed + near ground.
    # Each factor is smooth and continuous, avoids binary collapse.
    leg_contact_both = min(nleft_contact, nright_contact)  # 0 or 1, mostly
    slow_factor = 1.0 / (1.0 + CONTACT_PROXY_K * speed)
    near_ground_factor = 1.0 / (1.0 + 4.0 * abs(ny_pos))
    # Geometric mean of three factors to avoid product collapse
    contact_proxy = (leg_contact_both * slow_factor * near_ground_factor) ** (1.0 / 3.0)
    soft_landing_bonus = 2.0 * contact_proxy

    # 5. Action Efficiency (sparse penalty on engine use)
    # Action 2 = main engine, 1 and 3 = orientation engines.
    # We penalize any non-zero action, not just magnitude.
    is_engine_on = 0.0 if action == 0 else 1.0
    fuel_cost = -FUEL_COST_WEIGHT * is_engine_on

    # Assemble total reward
    total_reward = gated_proximity + velocity_penalty + soft_landing_bonus + fuel_cost

    # Components dict for introspection
    components = {
        "proximity_reward": gated_proximity,
        "velocity_penalty": velocity_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "fuel_cost": fuel_cost
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 设计理念
本奖励函数放弃了传统的“加性组件堆叠”，采用 **性能门控结构 (performance-gated structure)**。核心思路：让主进展（接近目标）只在姿态健康的条件下充分发挥作用，从而迫使飞行器在靠近目标的过程中主动控制姿态，而不是先快速冲向目标、再试图纠正姿态（这是过去失败的主因）。

## Selected Task Family
- **task_family:** navigation_goal_reaching  
- **dynamics_subtype:** goal_approach_and_soft_contact  
- **control_type:** discrete (4 actions)

## Selected Reward Roles
1. **goal_proximity_reward** — 核心驱动力（mandatory）。使用 squared distance 的 improvement delta，凸化后对远程进展和近程微调均有效。
2. **orientation_stability_gate** — 软健康门（mandatory）。将姿态误差作为 proximal reward 的乘性门控器，代替独立的二次惩罚。错误姿势会直接削弱接近奖励，迫使策略在接近时就调整好角度。
3. **soft_landing_velocity_penalty** — 条件速度约束（mandatory）。使用 soft hinge（仅在速度 > 5 m/s 触发）并按地面接近度加权，只在高危害区域起作用。
4. **contact_proxy (soft landing bonus)** — 腿接触联合奖励（conditional）。将双接触、低速、近地面三个连续因子按几何平均组合，提供每步稠密引导，避免乘积塌缩。
5. **action_efficiency_penalty** — 燃料使用惩罚（secondary）。稀疏形式（只在引擎激活时罚款 0.15），避免过早抑制推力。

## Role-to-Signal Mapping
- `goal_proximity`: `obs[0]` (x_pos), `obs[1]` (y_pos), `next_obs[0]`, `next_obs[1]`
- `orientation_gate`: `obs[4]`, `next_obs[4]` (body_angle)
- `velocity_penalty`: `next_obs[2]` (x_vel), `next_obs[3]` (y_vel), `next_obs[1]` (ny_pos)
- `contact_proxy`: `next_obs[6]` (left_contact), `next_obs[7]` (right_contact), speed from x/y vel
- `fuel_cost`: `action` (0=off, else on)

## Formula Operators Used
- **improvement_delta** on squared distance (convexified progress, avoids local plateaus)
- **soft_health_gate** (`angle_gate` exponential) to condition progress on stability
- **soft hinge** `max(0, speed - threshold)` for velocity penalty, safe region preserved
- **joint_condition_proxy** with geometric mean (`** (1/3)`) for soft landing, continuous factors
- **sparse action penalty** on discrete engine activation

## Excluded Roles & Reasons
- **terminal_success_bonus / early_failure_penalty** — `explicit_success_false` and `explicit_failure_false`. Info dict empty; no termination reason accessible to distinguish success from crash.
- **time_penalty** — no step count or clock signal available.
- **dense energy measure** — no thrust magnitude signal (only discrete actions).

## Why No Terminal Success/Failure
Environment card explicitly states `explicit_success_flag_available: false` and `explicit_failure_flag_available: false`. All termination mapping is ambiguous. The soft_landing_proxy serves as a continuous approximation of success, rewarding the pattern without needing endpoint labels.

## Responsibilities Left for Later Iterations
- Dynamic curriculum / phase-based gating (no training_progress)
- Stronger action smoothness (would need action history)
- Direct angle velocity damping (presently mild, could be added as quadratic)

## Expected Post-Training Observations
- **Success:** Shorter approach, settling with low speed and both legs near origin.
- **Failure risks:**
  - *Oscillation* around target if gating too strict; gate coefficient (ANGLE_GATE_K) can be reduced.
  - *Stalling at high altitude* if fuel_cost is too high; reduce to 0.1.
  - *Hard landings* if velocity hinge threshold too high; lower VEL_THRESH.
  - *One-leg landing* slipping through proxy — if persistent, increase CONTACT_PROXY_K or switch to multiplicative proxy with higher leg contact weight.