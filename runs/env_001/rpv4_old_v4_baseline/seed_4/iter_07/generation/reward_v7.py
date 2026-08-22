def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v7。

    v6→v7 修改（L2 结构性改动 — 单一语义：消除 per-step state reward 的悬停 exploit，
    同时恢复导航梯度）：

    问题诊断（v6）：
    v6 的 goal_proximity 依赖 `0.3 * exp(-dist/0.5)` 作为 per-step state reward，
    指数衰减使 dist>1.0 时梯度几乎为零。agent 不点火（fuel 4.1%）→ crash（68步）。
    所有 checkpoint 完全平坦——zero learning。

    同时 settling_stability 的 `0.2 + 2.5*prox_gate` baseline 在远处也对速度施加惩罚，
    进一步抑制导航动机。

    更深层根因：v2-v6 共享的 `delta + coeff*exp(-dist/tau)` 框架存在内在矛盾——
    per-step state reward 总量与 episode 长度正相关，必然产生悬停 exploit。
    v2 侥幸成功仅因 landing=4.0 暂时压制了悬停动机，v5 暴露了脆弱性。

    v7 方案（三项 L2 结构性改动）：
    1. goal_proximity: 从 state reward 改为 potential-based shaping。
       potential(d) = 5.0/(0.2+d)，reward = Φ(next) - Φ(current)。
       每步提供梯度，episode 总量 bounded（≈22 from dist=2→0），悬停零积累。
    2. settling_stability: 去除 0.2 baseline。multiplier = 3.0*prox_gate。
       远处≈无惩罚，近处强约束。让 agent 先自由导航，再近处减速。
    3. attitude_constraint: 去除 0.3 baseline。multiplier = 2.5*prox_gate。
       同理——远处不约束角度，近处严格要求姿态。

    landing_contact 保持 2.0/腿（v2 验证有效），fuel 保持 -0.05。
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

    # ========== Component A: goal_proximity (v7: potential-based) ==========
    # v7: 用 potential shaping 替代 state reward。
    # potential(d) = 5.0 / (0.2 + d):
    #   d=0   → 25.0
    #   d=0.5 →  7.14
    #   d=1.0 →  4.17
    #   d=2.0 →  2.27
    #   d=5.0 →  0.96
    # 奖励 = Φ(next) - Φ(current)，靠近目标为正，远离为负。
    # Episode 总量 bounded ≈ potential(0) - potential(initial)，与步数无关。
    # 悬停时 Φ(next)=Φ(current) → 奖励为 0，无 exploit 可能。
    def potential(dist):
        return 5.0 / (0.2 + dist)

    goal_proximity = potential(next_dist) - potential(current_dist)

    # ========== 邻近门控（多个组件共用）==========
    # prox_gate: dist=0→1.0, dist=0.2→0.5, dist=1.0→0.17, dist=2.0→0.09
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)

    # ========== Component B: settling_stability (v7: 零基线) ==========
    # v7: 去除 0.2 baseline。multiplier 从 0.2+2.5*prox_gate → 3.0*prox_gate。
    # 远处（dist>2, prox_gate≈0.09）：penalty ≈ 0.11/speed_unit，几乎不惩罚。
    # 近处（dist=0, prox_gate=1.0）：penalty ≈ 1.2/speed_unit，强减速。
    settling_multiplier = 3.0 * prox_gate
    settling_stability = -0.4 * next_speed * settling_multiplier

    # ========== Component C: attitude_constraint (v7: 零基线) ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)

    # v7: 去除 0.3 baseline。multiplier 从 0.3+2.0*prox_gate → 2.5*prox_gate。
    # 远处几乎不惩罚角度，近处严格要求姿态平稳。
    attitude_multiplier = 2.5 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.4 * next_angle_err + 0.1 * next_ang_vel_mag)

    # ========== Component D: landing_contact ==========
    # 保持 2.0/腿（v2 验证有效）。双足着陆 = 4.0/步。
    # 由于 goal_proximity 是 potential-based（悬停=0），不存在悬停收割竞争。
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