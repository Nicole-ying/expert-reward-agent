def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v6。

    v5→v6 修改（L1 双系数联动 — 单一语义：消灭悬停陷阱）：

    问题诊断（v5）：
    v5 将 landing 从 0.5→1.0/腿，意图恢复「着陆 > 盘旋」梯度（2.0 vs 1.0）。
    结果 score 从 172.9 暴跌至 -5.9，20/20 截断，landing 完全死亡。

    根因不是 landing 系数太小，而是 proximity_bonus = 1.0 * exp(-dist/0.5)
    作为无上限的 per-step 状态奖励，悬停 1000 步可收割 ~370-1000，
    远超 landing 可获取总量（即使双足着陆 4.0/步 × ~50 步 = 200）。
    agent 在 cp60% 学会着陆后（landing=94），cp80%-100% 发现了更优的悬停策略，
    于是完全放弃着陆 → landing 归零，所有 episode 跑满 1000 步截断。

    证据链：
    - goal_proximity ep_sum 与 episode 长度完全正相关：v2(404步)=193 → v4(663)=391 → v5(1000)=643
    - v5 checkpoint: cp60% landing=94.1 → cp80%=48.4 → cp100%=0（典型的策略替换）
    - v2（landing=2.0, proximity=1.0, 4:1 比）在 178.7 正常工作

    v6 方案（L1 双系数联动）：
    1. proximity_bonus: 1.0 → 0.3（削弱悬停的每步收益）
    2. landing_contact: 1.0/腿 → 2.0/腿（恢复 v2 验证有效的着陆激励）
    
    着陆 max 4.0/步 vs 悬停 proximity 0.3/步 = 13:1 梯度比，
    远超 v2 的 4:1，安全边际充足。其余组件不变。

    预期：landing 恢复活跃，agent 主动着陆终结 episode，len 回落至 400-500，
    score → 185-195。
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

    # ========== Component A: goal_proximity ==========
    # 改进量（保持 credit assignment 一致性，不受悬停影响）
    dist_improvement = current_dist - next_dist

    # v6: 系数从 1.0 → 0.3，削弱悬停的每步收益
    # exp(-dist/0.5): dist=0→0.3, dist=0.5→0.11, dist=1.0→0.04
    # 悬停 1000 步最多收割 300（vs v5 的 1000），不再能淹没 landing 信号
    proximity_bonus = 0.3 * (2.718281828 ** (-next_dist / 0.5))

    goal_proximity = 3.0 * dist_improvement + proximity_bonus

    # ========== 邻近门控（多个组件共用）==========
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)

    # ========== Component B: settling_stability ==========
    # 保持不变：速度惩罚，邻近目标时增强
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.4 * next_speed * settling_multiplier

    # ========== Component C: attitude_constraint ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)

    # 保持不变：角度 + 角速度惩罚，邻近目标时增强
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.4 * next_angle_err + 0.1 * next_ang_vel_mag)

    # ========== Component D: landing_contact (v6: 回归 2.0/腿) ==========
    # v6: 系数从 1.0/腿 → 2.0/腿。回归 v2 验证有效的设置。
    # 双足着陆 = 4.0/步 vs proximity_bonus 0.3/步 = 13:1 梯度比。
    # agent 有极强动机完成着陆并终结 episode，而非悬停收割 proximity。
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