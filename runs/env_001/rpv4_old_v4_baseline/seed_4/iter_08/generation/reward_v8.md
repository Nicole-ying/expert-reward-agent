Now I have the full picture. Let me do the systematic diagnosis.

---

## Systematic Component-by-Component Diagnosis

### Historical arc (v1→v7):

| iter | goal_proximity formula | score | len | fate |
|------|----------------------|-------|-----|------|
| v1 | pure delta | -121.5 | 68 | crash (no navigation signal) |
| v2 | `3.0*Δ + 1.0*exp(-d/0.5)` | **178.7** | 404 | **BEST** — landed successfully |
| v3 | same as v2, landing→transition | 88.5 | 892 | bounce exploit |
| v4 | same as v2, landing 0.5/leg | 172.9 | 663 | near best, slower |
| v5 | same as v2, landing 1.0/leg | -5.9 | 1000 | hover dominates |
| v6 | `3.0*Δ + 0.3*exp(-d/0.5)` | -115.6 | 68 | crash — exp gradient too weak |
| v7 | potential-based `Φ(next)-Φ(cur)` | -122.8 | 68 | crash — zero gradient at distance |

### Root cause analysis:

The v2-v6 family all shared the same `dist_improvement + coeff*exp(-dist/tau)` structure. This creates an **inherent tradeoff**: strong `coeff` → navigation works but hover exploit possible; weak `coeff` → no hover but navigation dies. v7 tried potential-based shaping and also failed because the potential gradient `5.0/(0.2+d)` flattens at distance.

**The fundamental unmet need**: a navigation signal that (a) provides uniform gradient at all distances, (b) cannot be exploited by hovering. The `exp(-dist/tau)` state reward fails (a) when coefficient is small, and fails (b) when coefficient is large. Potential shaping fails (a) due to 1/d decay.

### What has NOT been tried:

**Velocity-direction reward.** Instead of rewarding "being near the target" (state reward), reward "moving toward the target" (action-aligned reward). This provides gradient everywhere, is zero when hovering, and is naturally bounded per episode by total path length.

This is a **L2 change** to `goal_proximity` — new mathematical form, single semantic direction.

### Plan for other components:

- **settling_stability**: Keep v2's baseline framework `0.2 + 2.5*prox_gate` — v2 proved the baseline is essential for speed management without suppressing navigation.
- **attitude_constraint**: Keep v2's `0.3 + 2.0*prox_gate` — same logic.
- **landing_contact**: Keep v2's `2.0/leg` — with velocity-direction reward (no hover exploit possible), the landing signal is clean.
- **fuel_efficiency**: Keep `-0.05`.

This is materially different from all 7 historical iterations — none used velocity projection.

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v8。

    v7→v8 修改（L2 结构性改动 — 单一语义：用速度方向奖励替代状态邻近奖励）：

    问题诊断（v2-v7 全系列）：
    v2-v6 共享 `dist_improvement + coeff*exp(-dist/tau)` 框架，存在内在矛盾：
    - coeff 大 (v2=1.0)：导航有效但悬停可收割（v5 暴露，1000步截断）
    - coeff 小 (v6=0.3)：悬停消除但导航梯度死亡（68步 crash）
    v7 的 potential-based shaping `5.0/(0.2+d)` 在远处梯度也趋于零 → 68步 crash。

    根本原因：所有历史版本都用「距离状态的函数」作为导航奖励，
    这不可避免地在远处梯度弱化（指数/倒数衰减）。

    v8 方案：用「速度在目标方向上的投影」替代 exp(-dist/tau)。
    - vel_toward = -(v · pos) / |pos|：速度向量在指向原点方向上的投影
    - 正值 = 正在靠近目标，负值 = 正在远离
    - 梯度均匀（不随距离衰减）：dist=5 和 dist=0.5 处同样速度产生同样奖励
    - 悬停免疫：hovering → v≈0 → vel_toward≈0，零收割
    - 总量天然 bounded：总奖励 ≈ 系数 × 总路径长度，与 episode 长度无关

    其余组件保持 v2 验证有效的基线框架（settling/attitude 带 baseline，
    landing per-step 2.0/腿），因为 v2 证明这些设置可以完成任务。
    """

    # ========== 信号提取 ==========
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # ========== 距离与速度 ==========
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    next_speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5

    # ========== Component A: goal_proximity (v8: delta + velocity-direction) ==========
    # 距离改进量：每步靠近量，总量 bounded ≈ initial_dist - final_dist
    dist_improvement = current_dist - next_dist

    # 速度在目标方向上的投影（v8 核心创新）
    # 单位方向向量 = (-pos_x, -pos_y) / |pos|，指向原点（目标）
    # vel_toward = dot(velocity, direction_to_target)
    #            = (v_x, v_y) · (-pos_x, -pos_y) / |pos|
    #            = -(v_x*pos_x + v_y*pos_y) / |pos|
    # 正值 → 正在靠近目标；负值 → 正在远离；≈0 → 横向运动或静止
    # 关键性质：梯度不随距离衰减，悬停时 ≈0（无收割可能）
    if next_dist > 0.01:
        vel_toward = -(next_x_vel * next_x_pos + next_y_vel * next_y_pos) / next_dist
    else:
        vel_toward = 0.0

    # dist_improvement 提供主要导航梯度（每步靠近量）
    # vel_toward 提供辅助方向引导，鼓励直线高效靠近（而非绕弯）
    # 系数 0.3：速度=3 直线靠近时贡献 ≈0.9/步，与 v2 的 exp(0)=1.0 量级相当
    goal_proximity = 3.0 * dist_improvement + 0.3 * vel_toward

    # ========== 邻近门控（多个组件共用）==========
    # prox_gate: dist=0→1.0, dist=0.5→0.29, dist=1.0→0.17, dist=2.0→0.09
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)

    # ========== Component B: settling_stability (保持 v2 基线框架) ==========
    # 基线 0.2 确保远处也有最低速度约束
    # 远处: multiplier≈0.2+0.23=0.43, 近处: multiplier≈0.2+2.5=2.7
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.3 * next_speed * settling_multiplier

    # ========== Component C: attitude_constraint (保持 v2 基线框架) ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)

    # 基线 0.3 确保远处也有最低姿态约束
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.3 * next_angle_err + 0.1 * next_ang_vel_mag)

    # ========== Component D: landing_contact (v2 验证有效) ==========
    # 目标附近 + 低速时，双足着陆 = 4.0/步
    # 由于 goal_proximity 不再有 per-step state reward（悬停时 vel_toward≈0），
    # landing 信号干净无竞争
    near_target = next_dist < 0.5
    low_speed = next_speed < 1.0

    landing_contact = 0.0
    if near_target and low_speed:
        landing_contact = 2.0 * (next_left_contact + next_right_contact)

    # ========== Component E: fuel_efficiency ==========
    fuel_efficiency = -0.05 * float(action != 0)

    # ========== 汇总 ==========
    total_reward = (
        goal_proximity +
        settling_stability +
        attitude_constraint +
        landing_contact +
        fuel_efficiency
    )

    components = {
        'goal_proximity': goal_proximity,
        'settling_stability': settling_stability,
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency
    }

    return float(total_reward), components
```