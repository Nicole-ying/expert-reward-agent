def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    假设环境观测结构:
    obs[0]: x_position_relative_to_target
    obs[1]: y_position_relative_to_pad_height
    obs[2]: x_velocity
    obs[3]: y_velocity
    obs[4]: body_angle
    obs[5]: angular_velocity
    obs[6]: left_contact_flag (0.0/1.0)
    obs[7]: right_contact_flag (0.0/1.0)
    """
    # ---- 1. Progress reward: 距离减少量 (密集主学习信号) ----
    def distance(obs_arr):
        return (obs_arr[0]**2 + obs_arr[1]**2 + 1e-8) ** 0.5

    dist_old = distance(obs)
    dist_new = distance(next_obs)
    progress_reward = dist_old - dist_new

    # ---- 2. Speed tracking reward: 期望速度引导，代替硬惩罚 ----
    # 期望速度大小与距离成线性（远快近慢），用于引导合理的飞行速度
    max_speed = 5.0          # 预设的最大期望速度
    d_ref = 1.0              # 参考距离，在此距离下期望速度达到 max_speed
    desired_speed = max_speed * min(dist_new / d_ref, 1.0)

    cur_speed = (next_obs[2]**2 + next_obs[3]**2 + 1e-8) ** 0.5
    speed_error = abs(cur_speed - desired_speed)
    lambda_speed = 0.2
    speed_tracking_reward = -lambda_speed * speed_error

    # ---- 3. Soft landing proxy: 多条件线性组合，避免乘积梯度塌缩 ----
    proximity_threshold = 0.5
    speed_threshold = 0.25
    angle_threshold = 0.2     # 弧度

    proximity_score = max(0.0, 1.0 - dist_new / proximity_threshold)
    speed_low_score = max(0.0, 1.0 - cur_speed / speed_threshold)
    angle_score = max(0.0, 1.0 - abs(next_obs[4]) / angle_threshold)
    contact_score = (next_obs[6] + next_obs[7]) * 0.5

    # 线性组合，接触条件权重略高
    soft_landing_proxy = (
        proximity_score +
        speed_low_score +
        angle_score +
        2.0 * contact_score
    )  # 最大可能值约为 5.0

    # ---- 总奖励 ----
    total_reward = progress_reward + speed_tracking_reward + soft_landing_proxy

    components = {
        'progress_reward': progress_reward,
        'speed_tracking_reward': speed_tracking_reward,
        'soft_landing_proxy': soft_landing_proxy
    }
    return float(total_reward), components