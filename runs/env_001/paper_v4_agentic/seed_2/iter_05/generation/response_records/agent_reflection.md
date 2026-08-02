# Response Record

# 设计理由

本轮修改一个组件：将 `landing_bonus`（僵尸组件，active_rate=0%）替换为 `soft_approach_bonus`。原组件依赖腿接触标志与严格速度阈值，从未被激活；agent 的终止模式表明它在高速撞击中 crash，没有机会产生腿接触或满足窄阈值。新组件去除腿接触依赖，使用连续的 y 高度 gate 与速度、角度因子，在接近着陆垫低高度且速度、姿态良好时给予正奖励，从而提供可学习的软着陆梯度。数学形式为三因子的线性 hinge 乘积，以保证在初始随机探索阶段就有一定激活概率。系数选择使单步最大奖励约为 10，预期 episode sum 与主信号 `proximity_delta` 相当或稍低，不破坏整体平衡。velocity_danger 与 orientation_penalty 保持不变。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v5 — replace dead landing_bonus with soft_approach_bonus using y distance gate
    and continuous velocity/angle factors. No contact dependency.
    """
    # ── current state ──
    x_cur = obs[0]
    y_cur = obs[1]
    vx_cur = obs[2]
    vy_cur = obs[3]
    angle_cur = obs[4]
    angvel_cur = obs[5]

    # ── next state ──
    x_next = next_obs[0]
    y_next = next_obs[1]

    # ── distance to pad (current) ──
    dist_cur  = (x_cur  ** 2 + y_cur  ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # ── weights / thresholds ──
    w_prox = 50.0
    w_vel  = 0.15
    w_ang  = 5.0
    proximity_threshold = 1.0

    w_approach = 10.0               # moderate bonus for near-ground stability

    # ── 1. Proximity delta ──
    proximity_delta = w_prox * (dist_cur - dist_next)

    # ── 2. Velocity danger ──
    speed_sq = vx_cur ** 2 + vy_cur ** 2
    velocity_danger = -w_vel * speed_sq / (dist_cur + proximity_threshold)

    # ── 3. Orientation penalty ──
    orientation_penalty = -w_ang * (angle_cur ** 2 + angvel_cur ** 2)

    # ── 4. Soft approach bonus (replaces landing_bonus) ──
    # Gate: how close we are to the pad vertically (y near 0)
    closeness = max(0.0, 1.0 - abs(y_next) / 0.5)   # active when |y| < 0.5

    # Speed factor: total speed should be low
    total_speed = (vx_cur ** 2 + vy_cur ** 2) ** 0.5
    speed_factor = max(0.0, 1.0 - total_speed / 1.0)   # active when speed < 1.0

    # Angle factor: body should be upright
    angle_factor = max(0.0, 1.0 - abs(angle_cur) / 0.5)  # active when |angle| < 0.5

    soft_approach_bonus = w_approach * closeness * speed_factor * angle_factor

    # ── Total reward ──
    total_reward = proximity_delta + velocity_danger + orientation_penalty + soft_approach_bonus

    components = {
        "proximity_delta": proximity_delta,
        "velocity_danger": velocity_danger,
        "orientation_penalty": orientation_penalty,
        "soft_approach_bonus": soft_approach_bonus,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测维度已被使用，无信号缺口；`landing_bonus` 依赖的腿接触与严格速度阈值导致 active_rate=0%，是僵尸组件，需以连续因子替代。
- **behavior**: agent 以高速撞击式 crash 结束，平均长度仅 68 步，从未产生安全着陆腿接触。
- **signal**: 缺少可学习的接近地面时减速对齐的正向密集信号；原有惩罚不足以抑制高速。
- **level**: Level 2 — 将二值/条件严格、从未激活的 `landing_bonus` 替换为基于 y 高度 gate 的连续有界 soft_approach_bonus。
- **hypothesis**: 新组件在低高度时奖励低速与正姿态，提供可微分梯度引导 agent 在接近着陆垫时减速并调姿，从而延长 episode 并实现软着陆。
- **risk**: 若总速度与姿态仍难以达到阈值范围，soft_approach_bonus 初期激活率仍可能偏低，但线性 hinge 的宽阈值（speed<1.0, angle<0.5, |y|<0.5）应显著高于原组件，且 `velocity_danger` 配合施加了减速压力。
