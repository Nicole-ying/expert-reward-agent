def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ====== 主学习信号：前进速度奖励 ======
    # 直接使用下一步的水平线速度，鼓励快速向前，后退自然为负
    forward_vel = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_vel

    # ====== 竖直/姿态稳定性惩罚 ======
    # 保持躯干直立、抑制大幅摇摆和垂直跳动
    hull_angle = obs[0]           # 躯干倾角
    hull_angvel = obs[1]          # 躯干角速度
    vertical_vel = obs[3]         # 垂直速度
    w_angle = 0.5
    w_angvel = 0.01
    w_vertical = 0.01
    stability_penalty = -(
        w_angle * hull_angle**2 +
        w_angvel * hull_angvel**2 +
        w_vertical * vertical_vel**2
    )

    # ====== 能量效率惩罚 ======
    # 降低关节力矩幅值，体现能耗目标
    w_energy = 0.001
    energy_penalty = -w_energy * (
        action[0]**2 + action[1]**2 +
        action[2]**2 + action[3]**2
    )

    total_reward = forward_reward + stability_penalty + energy_penalty
    components = {
        'forward_reward': forward_reward,
        'stability_penalty': stability_penalty,
        'energy_penalty': energy_penalty
    }
    return float(total_reward), components