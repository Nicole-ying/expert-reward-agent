# Response Record

---

## 诊断报告

### evidence

| 信号 | iter_3 数据 | 解读 |
|---|---|---|
| **Score** | −65.0，cp80% 时最优 −24.3，之后退化到 −65.3 | 经典 exploitation：agent 先学会靠近，然后发现「中等距离悬停」更划算 |
| **Episode** | len=1000，truncated=20/20，terminated=0/20 | agent 活满整集但从不自然终止——从未成功着陆 |
| **goal_proximity** | ep_sum=1204，magnitude_share=**86%**，cp80%→cp100% 从 1625 降到 1203 | `3.0*e^(-dist)` 永久正奖励，agent 悬停在 ~0.9 距离收租，后退后还在退化 |
| **landing_contact** | **0.0 / 0.0% active**，全程所有 checkpoint 均为 0 | 死组件。dist<0.25 且 speed<0.4 的门控太紧，agent 从未触发 |
| **attitude_constraint** | −89.1，cp80% 时 −117（最负），cp100% 时恢复到 −90 | agent 靠近后姿态惩罚加重，于是主动后退以减轻惩罚 |
| **settling_stability** | −34.7，cp80% −46.5 → cp100% −34.3 | 同上：后退减轻了速度惩罚 |
| **fuel_efficiency** | −42.5，active 85%，cp 间持续恶化 | agent 引擎使用率上升（悬停需要频繁点火维持位置） |

### behavior_diagnosis

**学到的行为**：agent 学会了「靠近但不降落」——在距离 ~0.9 处悬停 1000 步，收集 `3.0*e^(-0.9)≈1.2/步` 的 proximity 奖励，同时避免触发 attitude_constraint（gate 在 dist<2.0 时激活）和 settling_stability（prox_gate 随距离减小而增大）的强惩罚。

**退化机制**：cp80% 时 agent 一度靠得更近（goal_proximity=1625，对应平均 dist≈0.6），但 attitude_constraint 罚到 −117、settling 罚到 −46.5。agent 发现后退到 dist≈0.9 时这些惩罚大幅减轻（attitude→−90，settling→−34），而 proximity 仍能收 ~1.2/步，**净收益更高**。于是策略退化。

**根本原因**：
1. `proximity_bonus = 3.0 * e^(-dist)` 是**无界累积**的正向状态奖励——只要活着就能收租，导致 86% 的 reward 来自悬停
2. landing_contact 门控过紧（dist<0.25 + speed<0.4），agent 在悬停距离 (~0.9) 根本碰不到
3. 惩罚组件（attitude、settling）在目标附近变强，反而**惩罚着陆**——agent 靠近目标后净收益下降

### 干预策略

**核心思路**：将 `goal_proximity` 从「永久正向状态奖励」改为**纯势能差分（potential-based shaping）**——`reward = Φ(next) - Φ(current)`，使总奖励仅取决于起点和终点，无法通过悬停/振荡剥削。同时扩大 landing_contact 门控、增大着陆奖励，让「着陆」成为唯一最优策略。

具体改动：
1. **goal_proximity**：移除 `3.0*e^(-dist)` 的永久 bonus，改用势能差分 `Φ = -w1*dist + w2*e^(-dist)`，reward = Φ(next)−Φ(current)。总奖励有界（≈w1*initial_dist + w2），悬停收益为 0
2. **settling_stability**：降低状态惩罚系数，让靠近目标不再痛苦
3. **attitude_constraint**：降低姿态惩罚系数，扩宽门控死区
4. **landing_contact**：放宽门控（dist<0.4, speed<0.6），提高奖励至 30.0，增加渐进激励

---

## v4 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v4。

    根本性重构（vs v3）：
    - goal_proximity：从「永久正 state bonus + delta」切换为
      **纯势能差分 potential-based shaping**。
      Φ(s) = -w_lin * dist + w_exp * e^(-dist)
      reward = Φ(next_obs) - Φ(obs)
      总奖励有界（≈ w_lin * initial_dist + w_exp），悬停收益=0，振荡净收益=0。
      彻底消除 v3 中 agent 悬停收租的 exploit。

    - settling_stability：降低状态惩罚系数（-1.0 vs v3 的 -1.2），
      让 agent 不再因靠近目标而被惩罚到退缩。

    - attitude_constraint：降低姿态/角速度惩罚系数，扩宽门控范围
      （dist>=2.5 死区 vs v3 的 2.0），让中距离机动不被惩罚。

    - landing_contact：放宽门控（dist<0.4 vs v3 的 0.25，speed<0.6 vs v3 的 0.4），
      提高双支撑奖励至 30.0，让着陆信号真正可触达且足够诱人。
    """

    # ========== 信号提取 ==========
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # L2 距离与速率
    current_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    current_speed = (vx**2 + vy**2) ** 0.5
    next_speed = (nvx**2 + nvy**2) ** 0.5

    # ================================================================
    # Component A: goal_proximity — 纯势能差分 (potential-based shaping)
    #
    #   Φ(s) = -w_lin * dist + w_exp * e^(-dist)
    #   reward = Φ(next) - Φ(current)
    #
    #   数学性质：
    #   - 总 episode 奖励 = Φ(final) - Φ(initial)，路径无关
    #   - 悬停 (dist 不变)：reward = 0
    #   - 振荡 (来回)：净 reward = 0
    #   - 靠近目标：reward > 0；远离：reward < 0
    #
    #   系数设计：
    #   - w_lin=5.0：线性梯度，远处每步 ~0.5-1.0 的正向信号
    #   - w_exp=5.0：指数势能，在目标附近提供陡峭梯度
    #     dist=0→Φ_exp=5.0, dist=0.5→Φ_exp=3.03, dist=1→Φ_exp=1.84
    # ================================================================
    phi_current = -5.0 * current_dist + 5.0 * (2.718281828 ** (-current_dist))
    phi_next    = -5.0 * next_dist    + 5.0 * (2.718281828 ** (-next_dist))
    goal_proximity = phi_next - phi_current

    # ================================================================
    # Component B: settling_stability — 门控速度管理
    #
    #   比 v3 更轻的状态惩罚（-1.0 vs -1.2），降低靠近目标时的阻力。
    #   prox_gate: dist=0→1.0, dist=1→0.33, dist=2→0.20
    # ================================================================
    prox_gate = 1.0 / (1.0 + 2.0 * next_dist)
    speed_improvement = current_speed - next_speed

    # delta：奖励减速，近处权重放大
    speed_delta = 2.5 * speed_improvement * (0.5 + 2.0 * prox_gate)

    # 状态惩罚：轻量线性速度惩罚，仅在接近目标时生效
    speed_state = -1.0 * next_speed * prox_gate

    settling_stability = speed_delta + speed_state

    # ================================================================
    # Component C: attitude_constraint — 扩宽死区的姿态管理
    #
    #   dist >= 2.5：完全不干预（v3 是 2.0），给中距离机动更多自由
    #   dist <= 0.5：全力约束
    #   中间线性过渡
    #   降低惩罚系数以减少靠近目标时的排斥力
    # ================================================================
    if next_dist >= 2.5:
        att_gate = 0.0
    elif next_dist <= 0.5:
        att_gate = 1.0
    else:
        att_gate = (2.5 - next_dist) / 2.0  # 从 1.0 线性降到 0.0

    angle_improvement = abs(angle) - abs(n_angle)
    angvel_improvement = abs(ang_vel) - abs(n_ang_vel)

    angle_delta = att_gate * (2.0 * angle_improvement + 1.0 * angvel_improvement)
    # 降低状态惩罚系数：angle: -2.0(v4) vs -3.0(v3), angvel: -1.0(v4) vs -1.5(v3)
    angle_penalty = att_gate * (-2.0 * abs(n_angle) - 1.0 * abs(n_ang_vel))

    attitude_constraint = angle_delta + angle_penalty

    # ================================================================
    # Component D: landing_contact — 核心正向激励（放宽门控）
    #
    #   放宽条件让着陆信号可触达：
    #   - dist < 0.4（v3 是 0.25）
    #   - speed < 0.6（v3 是 0.4）
    #   提高双支撑奖励至 30.0（v3 是 25.0）
    #   单边接触给 6.0 作为渐进引导
    # ================================================================
    near_target = next_dist < 0.4
    low_speed = next_speed < 0.6
    both_contacts = (n_left > 0.5 and n_right > 0.5)
    any_contact = (n_left > 0.5 or n_right > 0.5)

    landing_contact = 0.0
    if near_target and low_speed:
        if both_contacts:
            landing_contact = 30.0
        elif any_contact:
            landing_contact = 6.0

    # ================================================================
    # Component E: fuel_efficiency — 轻量引擎使用惩罚
    # ================================================================
    fuel_efficiency = -0.05 * float(action != 0)

    # ================================================================
    # 汇总
    # ================================================================
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
        'fuel_efficiency': fuel_efficiency,
    }

    return float(total_reward), components
```
