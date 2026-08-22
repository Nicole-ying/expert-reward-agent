def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v5。
    
    核心改动：landing_contact 从持续状态奖励 → 状态转移奖励。
    - sustain: 小额维持奖励（0.3x），防止信号完全消失但不再鼓励徘徊
    - transition: 大额转移奖励（4.0x），仅在接触程度增加时发放
    - 使用 obs 和 next_obs 的接触状态差来检测转移事件
    
    设计原则：
    - goal_proximity: 二次距离代价 → 引导接近（远处强、近处弱）
    - settling_stability: 二次速度代价 → 引导减速（高速强、低速弱）
    - landing_contact: 转移型接触奖励 → 奖励"取得接触"而非"保持接触"
    - attitude_constraint: 姿态代价 → 保持平稳
    - fuel_efficiency: 燃料代价 → 节省推力
    
    Args:
        obs: 当前观测 (8,)
        action: 当前动作 (0-3)
        next_obs: 下一观测 (8,)
        original_reward: 未使用
        info: 未使用
        training_progress: 未使用
    """
    
    # ========== 信号提取 ==========
    # 上一状态（用于检测转移）
    prev_left_contact = obs[6]
    prev_right_contact = obs[7]
    
    # 当前状态（转移后）
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # ========== 辅助量 ==========
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    
    # proximity_gate: 1.0 在目标处，~0.1 在 distance≈3
    proximity_gate = 1.0 / (1.0 + 3.0 * distance)
    
    # ========== Component A: goal_proximity — 二次距离代价 ==========
    # 梯度 = -0.2 * distance，远处梯度强，近处梯度弱（天然 curriculum）
    goal_proximity = -0.1 * distance ** 2
    
    # ========== Component B: settling_stability — 二次速度代价 ==========
    # 梯度随 speed 和 proximity 增强，接近目标时减速约束更强
    settling_stability = -0.3 * speed ** 2 * (0.2 + 0.8 * proximity_gate)
    settling_stability += -0.08 * ang_vel ** 2 * (0.2 + 0.8 * proximity_gate)
    
    # ========== Component C: attitude_constraint — 姿态代价 ==========
    angle_error = abs(body_angle)
    ang_vel_mag = abs(ang_vel)
    # gate 范围：远距离 0.3，近距离 1.0
    attitude_gate = 0.3 + 0.7 * proximity_gate
    attitude_constraint = -0.4 * angle_error * attitude_gate
    attitude_constraint += -0.15 * ang_vel_mag * attitude_gate
    
    # ========== Component D: landing_contact — 转移型着陆奖励 ==========
    # v5 核心改动：从持续状态奖励改为状态转移奖励。
    # - sustain: 小额维持信号（系数 0.3），不再鼓励在接触区徘徊
    # - transition: 大额转移信号（系数 4.0），强烈奖励"取得/增加接触"
    # 这防止了 agent 在 near_pad+slow_enough 区域内无限徘徊领取 contact 奖励。
    prev_contact_frac = (prev_left_contact + prev_right_contact) / 2.0
    curr_contact_frac = (left_contact + right_contact) / 2.0
    contact_gain = max(0.0, curr_contact_frac - prev_contact_frac)  # 仅奖励接触增加
    
    near_pad = distance < 0.5
    slow_enough = speed < 1.0
    
    if near_pad and slow_enough:
        prox_factor = 1.0 - distance / 0.5      # 1.0@d=0, 0.0@d=0.5
        speed_factor = 1.0 - speed / 1.0        # 1.0@s=0, 0.0@s=1.0
        
        # 维持奖励：微弱正向信号，保持接触状态的价值感知
        sustain = 0.3 * prox_factor * speed_factor * curr_contact_frac
        # 转移奖励：接触程度增加时的大额奖励，驱动"完成着陆"这个动作
        transition = 4.0 * prox_factor * speed_factor * contact_gain
        
        landing_contact = sustain + transition
    else:
        landing_contact = 0.0
    
    # ========== Component E: fuel_efficiency — 燃料代价 ==========
    fuel_efficiency = -0.03 * float(action != 0)
    
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