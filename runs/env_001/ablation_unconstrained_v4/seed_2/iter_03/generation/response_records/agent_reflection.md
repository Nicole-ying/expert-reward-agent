# Response Record

基于训练反馈，agent 所有 episode 均因步数上限截断（1000步），从未触发成功着陆或坠毁终止。奖励分解显示，水平位置惩罚（horizontal_penalty）占总惩罚的86.7%，但其逐步平均值很小（对应 x≈0.15），表明 agent 已学会在水平方向上贴近中心，但几乎静止不下降。progress 仅1.19（每步约0.001），说明 agent 几乎没有向目标点移动，垂直进展几乎为零。contact_bonus 激活率为0，因为 agent 从未触地。这些现象说明：当前奖励结构下，agent 最优策略是保持水平居中、悬停不动以规避所有惩罚，而 progress 信号过弱不足以驱动下降。核心问题是**下降激励不足、水平约束形态不当（全时惩罚抑制探索）、着陆引导缺乏**。

修改方向：降低水平惩罚权重并转为主奖励的辅助因子，强化下降过程的稠密激励，并在接近地面时引入速度质量门控，使 agent 在下降的同时学会减速着陆。同时保留姿态/速度安全约束，并将接触奖励放宽为任一支撑腿接触即可获取连续信号。

修改实施：
- 提高 progress 权重至 5.0，使其成为主要正信号源。
- 新增垂直下降奖励 `y_progress`，直接奖励高度减小，权重 2.0。
- 水平对齐改为正奖励 `alignment`（接近 0 时高奖励），替代全时惩罚，仅保留极小的水平惩罚（权重 0.05）防止过分飘远。
- 接触奖励改为任意支撑腿接触即可激活，并乘以姿态与落速质量因子，权重 3.0。
- 添加接近地面奖励 `height_near_reward`，当高度 <0.5 时提供线性激励，引导触地。
- 安全约束保持低权重（0.1），避免压制探索。
- 返回所有组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前与下一状态
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new, vy_new = next_obs[2], next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 到原点的距离（目标平台位于 (0,0)）
    dist_old = (x_old ** 2 + y_old ** 2) ** 0.5
    dist_new = (x_new ** 2 + y_new ** 2) ** 0.5

    # 权重
    w_progress = 5.0
    w_y_progress = 2.0
    w_alignment = 0.5
    w_horizontal_penalty = 0.05
    w_vel = 0.1
    w_att = 0.1
    w_contact = 3.0
    w_ground = 1.0

    # 阈值
    vx_max = 0.5
    vy_max = 0.5
    angle_max = 0.2
    angvel_max = 0.2
    ideal_angle_tol = 0.1
    ideal_vy_tol = 0.2

    # 1. 向目标移动的稠密进展奖励（距离缩短）
    progress = w_progress * (dist_old - dist_new)

    # 2. 垂直下降奖励（直接奖励高度减小）
    y_progress = w_y_progress * (y_old - y_new)

    # 3. 水平对准奖励（接近 x=0 时高奖励）
    alignment = w_alignment * (1.0 / (1.0 + x_new ** 2))

    # 4. 微小水平偏离惩罚（仅在需要时提供软边界）
    horizontal_penalty = -w_horizontal_penalty * (x_new ** 2)

    # 5. 安全速度约束
    excess_vx = max(0.0, abs(vx_new) - vx_max)
    excess_vy = max(0.0, abs(vy_new) - vy_max)
    vel_penalty = -w_vel * (excess_vx ** 2 + excess_vy ** 2)

    # 6. 姿态安全约束
    excess_angle = max(0.0, abs(angle_new) - angle_max)
    excess_angvel = max(0.0, abs(angvel_new) - angvel_max)
    att_penalty = -w_att * (excess_angle ** 2 + excess_angvel ** 2)

    # 7. 接触奖励（任意支撑腿接触 + 质量因子）
    any_contact = max(left_contact, right_contact)
    angle_quality = max(0.0, 1.0 - abs(angle_new) / ideal_angle_tol)
    vy_quality = max(0.0, 1.0 - abs(vy_new) / ideal_vy_tol)
    contact_reward = w_contact * any_contact * angle_quality * vy_quality

    # 8. 接近地面奖励（当高度低于阈值时线性激励）
    ground_threshold = 0.5
    height_near_reward = w_ground * max(0.0, ground_threshold - y_new)

    # 总和
    total_reward = (progress + y_progress + alignment + horizontal_penalty +
                    vel_penalty + att_penalty + contact_reward + height_near_reward)

    components = {
        'progress': progress,
        'y_progress': y_progress,
        'alignment': alignment,
        'horizontal_penalty': horizontal_penalty,
        'vel_penalty': vel_penalty,
        'att_penalty': att_penalty,
        'contact_reward': contact_reward,
        'height_near_reward': height_near_reward
    }

    return float(total_reward), components
```
