def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v1。
    核心目标：尽快到达中心目标垫板并稳定停靠，同时鼓励双支撑腿安全着陆。
    
    Args:
        obs: 当前观测 (8,) [x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, left_contact, right_contact]
        action: 当前动作 (0-3: no_engine, left_orient, main_engine, right_orient)
        next_obs: 下一观测 (8,)
        original_reward: 原始奖励（不使用）
        info: 信息字典（为空，不使用）
        training_progress: 训练进度（v1不使用）
    
    Returns:
        float: 总奖励
        dict: 各奖励组件
    """
    
    # ========== 信号提取 ==========
    # 当前状态
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    # 下一状态
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    # ========== Component A: 主学习信号 - 目标接近与稳定 ==========
    # role: goal_proximity + settling_stability
    # 策略：使用 improvement_delta 鼓励朝目标方向移动并减速
    # 水平方向：减少 |x_pos|，理想为 0（对准垫板中心）
    # 垂直方向：减少 |y_pos|，理想为 0（贴齐垫板表面）
    # 同时鼓励速度归零（settling），权重随接近目标而升高
    
    # 当前位置距离（L1范数，简单有效）
    current_distance = abs(x_pos) + abs(y_pos)
    next_distance = abs(next_x_pos) + abs(next_y_pos)
    
    # 位置改进奖励：距离减少得正奖励（improvement_delta）
    distance_improvement = current_distance - next_distance
    
    # 速度惩罚：鼓励减速，尤其在接近目标时
    current_speed = abs(x_vel) + abs(y_vel)
    next_speed = abs(next_x_vel) + abs(next_y_vel)
    
    # 速度改进：速度降低得奖励（越接近目标，权重越高）
    # 使用 gating: 距离越近，速度惩罚越重要
    proximity_gate = 1.0 / (1.0 + 5.0 * current_distance)  # 距离远时≈0，距离0时≈1
    speed_improvement = current_speed - next_speed
    
    # 组合：位置改进为主驱动，速度改进为辅约束
    goal_proximity_reward = 2.0 * distance_improvement  # 主导航信号
    settling_reward = 0.5 * speed_improvement * (1.0 + proximity_gate * 3.0)  # 接近时加强速度控制
    
    # ========== Component B: 姿态稳定约束 ==========
    # role: upright_orientation (条件激活) + 部分 settling_stability
    # 仅当接近目标时激活角度惩罚，避免干扰大角度机动
    
    # 当前和下一姿态误差
    current_angle_error = abs(body_angle)
    next_angle_error = abs(next_body_angle)
    current_ang_vel_mag = abs(ang_vel)
    next_ang_vel_mag = abs(next_ang_vel)
    
    # 姿态改进：角度减小、角速度减小得奖励
    angle_improvement = current_angle_error - next_angle_error
    ang_vel_improvement = current_ang_vel_mag - next_ang_vel_mag
    
    # 仅当接近目标时施加姿态约束（y_pos < 0.5 且 distance < 1.0）
    attitude_gate = 1.0 if (abs(y_pos) < 0.5 and current_distance < 1.0) else 0.1
    attitude_reward = attitude_gate * (1.0 * angle_improvement + 0.5 * ang_vel_improvement)
    
    # ========== Component C: 安全着陆接触奖励 ==========
    # role: safe_landing_contact
    # 当接近目标、速度低、且双支撑腿接触时给予正向奖励
    # 使用 joint_condition_proxy 构造软接触成功信号
    
    # 接近目标条件
    near_target = abs(next_x_pos) < 0.2 and abs(next_y_pos) < 0.2
    
    # 低速条件（L1速度小于阈值）
    low_speed = (abs(next_x_vel) + abs(next_y_vel)) < 0.5
    
    # 双接触条件
    both_contacts = (next_left_contact > 0.5 and next_right_contact > 0.5)
    
    # 接触改进：从单边或无边到双接触
    contact_improvement = 0.0
    if both_contacts and not (left_contact > 0.5 and right_contact > 0.5):
        contact_improvement = 0.5  # 新获得双接触
    
    # 组合门控：仅当接近且低速时，接触才有意义
    if near_target and low_speed:
        landing_reward = 3.0 * float(both_contacts)  # 鼓励维持双接触
        landing_reward += 2.0 * contact_improvement  # 额外奖励首次双接触
    else:
        landing_reward = 0.0
    
    # ========== Component D: 燃料效率惩罚（轻量） ==========
    # role: fuel_efficiency
    # 对使用引擎施加轻微惩罚，避免无意义动作
    # action != 0 表示使用了引擎（离散动作空间）
    fuel_penalty = -0.05 * float(action != 0)
    
    # ========== 汇总奖励 ==========
    total_reward = (
        goal_proximity_reward +
        settling_reward +
        attitude_reward +
        landing_reward +
        fuel_penalty
    )
    
    components = {
        'goal_proximity': goal_proximity_reward,
        'settling_stability': settling_reward,
        'attitude_constraint': attitude_reward,
        'landing_contact': landing_reward,
        'fuel_efficiency': fuel_penalty
    }
    
    return float(total_reward), components