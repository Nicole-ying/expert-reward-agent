# Response Record

Now I'll write the v10 code. The single change is replacing the Lorentzian proximity function with a narrower exponential.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v10。

    v9→v10 修改（L2 结构性改动 — 单一语义：指数邻近函数替代 Lorentzian 以消除悬停 exploit）：

    问题诊断（v9）：
    v9 的 Lorentzian proximity_bonus = 1/(1+(d/2.0)²) 过宽：
    - d=2.0 → 0.500/步，d=3.0 → 0.308/步，d=5.0 → 0.138/步
    - checkpoint 铁证：landing_contact cp70%→90% 活跃（77→294），cp100% 归零
      而 goal_proximity 持续增长至 926。agent 发现悬停收割 proximity > 着陆，
      完全放弃 landing。
    - 这是典型的「占据好状态即可持续获奖」→「proxy 提高但外部任务不升」

    根因：Lorentzian 代数衰减太慢。v2 的 exp(-d/0.5) 虽然被 v9 诊断为"太窄"，
    但 v2=178.70 证明窄恰恰是安全机制——悬停在 d≥2 处无利可图。

    v10 方案：
    proximity_bonus = exp(-d/0.6)，τ=0.6 介于 v2 (0.5) 和理想中值之间：
    - d=0.0 → 1.000    d=0.5 → 0.435    d=1.0 → 0.189
    - d=2.0 → 0.036    d=3.0 → 0.007    d=5.0 → 0.0002
    - 比 v2 宽 20%（d=1 处 0.189 vs 0.135），确保初始探索有足够梯度
    - 比 v9 Lorentzian 窄 14 倍（d=2 处 0.036 vs 0.500），悬停无法获利
    - 近目标（d<0.5）处 bonus≈0.4-1.0，与 landing_contact 4.0/步 形成 4:1-10:1 压制比

    其余组件保持 v9 不变（settling 零基线、attitude 0.3 基线、landing 2.0/腿）。
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

    # ========== Component A: goal_proximity (v10: 指数邻近奖励 τ=0.6) ==========
    # 距离改进量：每步靠近量，总量 bounded ≈ initial_dist - final_dist
    dist_improvement = current_dist - next_dist

    # v10: exp(-d/0.6) — 回到 v2 验证有效的指数函数族，τ 从 0.5 微扩至 0.6。
    #   d=0.0 → 1.000    d=0.5 → 0.435    d=1.0 → 0.189
    #   d=2.0 → 0.036    d=3.0 → 0.007    d=5.0 → 0.0002
    # v9 Lorentzian 对比：d=2.0→0.500, d=3.0→0.308 — 宽 14-44 倍，导致悬停 exploit。
    # v2 exp(-d/0.5) 对比：d=2.0→0.018 — v10 在 d=2 处信号翻倍但仍微不足道。
    # 关键：d≥2 时 bonus≤0.036，悬停 1000 步最多收割 36，远不如着陆（~400）。
    tau = 0.6
    proximity_bonus = 1.0 * (2.718281828 ** (-next_dist / tau))

    goal_proximity = 3.0 * dist_improvement + proximity_bonus

    # ========== 邻近门控（多个组件共用）==========
    # prox_gate: dist=0→1.0, dist=0.5→0.286, dist=1.0→0.167, dist=2.0→0.091
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)

    # ========== Component B: settling_stability (保持 v9 零基线) ==========
    # 零基线：multiplier = 3.0*prox_gate。
    # 远处（dist=5, prox_gate≈0.038）：multiplier≈0.115，speed=3→penalty≈-0.10/步
    # 近处（dist=0, prox_gate=1.0）：multiplier=3.0，penalty 强约束
    settling_multiplier = 3.0 * prox_gate
    settling_stability = -0.3 * next_speed * settling_multiplier

    # ========== Component C: attitude_constraint (保持 v9 框架) ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)

    # 基线 0.3 确保远处也有最低姿态约束（量级小，不干扰导航）
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.3 * next_angle_err + 0.1 * next_ang_vel_mag)

    # ========== Component D: landing_contact (v2 验证有效) ==========
    # 目标附近 + 低速时，双足着陆 = 4.0/步
    # 在 dist=0 处：proximity_bonus≈1.0，landing=4.0，4:1 压制悬停
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
