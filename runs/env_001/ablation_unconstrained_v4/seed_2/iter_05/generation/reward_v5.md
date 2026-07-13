基于训练反馈，当前得分为-8.16，显著差于历史最佳71.06，并且全部episode以truncated结束（步长1000），没有任何接触激活。这说明agent学会了利用`alignment`这一无条件的稠密奖励在空中悬停（`alignment`贡献了95.7%的正奖励），完全忽略了下降和着陆任务。解决本轮失败的关键是打破这种悬停策略：**让`alignment`只有当着陆器下降到接近平台高度时才有效**，同时强化下降进度和接近地面的引导，使agent必须下降才能获得主要的正向奖励。

我将以历史最佳迭代（iter3）的稀疏信号为导向，恢复并改进奖励结构：加入高度门控的`alignment`，增大`y_progress`和`height_near_reward`的权重，添加水平速度惩罚以辅助对准，并保留原有的软约束。这样既能迫使agent优先下降，又能在接近地面时获得精确的横向对准引导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 读取状态 ----------
    x_old, y_old = obs[0], obs[1]
    x_new = next_obs[0]
    y_new = next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 辅助量 ----------
    dist_old = (x_old**2 + y_old**2)**0.5
    dist_new = (x_new**2 + y_new**2)**0.5
    any_contact = max(left_contact, right_contact)
    double_contact = left_contact * right_contact

    # ---------- 进度奖励（鼓励持续下降和靠近）----------
    w_progress = 5.0
    w_y_progress = 10.0
    progress = w_progress * (dist_old - dist_new)
    y_progress = w_y_progress * (y_old - y_new)

    # ---------- 高度门控的水平对准（关键修改）----------
    # 只有当 y_new < height_threshold 时，alignment 才会激活，
    # 且奖励随高度降低线性增大，迫使 agent 先下降再对准
    height_threshold = 0.8
    height_gate = max(0.0, 1.0 - y_new / height_threshold)  # y_new >= 0.8 时=0，y_new=0 时=1
    alignment_raw = 1.0 / (1.0 + x_new**2)
    w_alignment = 0.8
    alignment = w_alignment * alignment_raw * height_gate

    # ---------- 接近地面鼓励（无接触门槛）----------
    w_height_near = 4.0
    ground_threshold = 0.8   # 使用同一阈值，使奖励在下降至0.8以下时持续增加
    height_near_reward = w_height_near * max(0.0, ground_threshold - y_new)

    # ---------- 接触奖励 ----------
    w_any_contact = 5.0
    any_contact_reward = w_any_contact * any_contact

    # ---------- 双腿稳定着陆奖励 ----------
    w_stable = 20.0
    vx_tol, vy_tol, angle_tol, angvel_tol = 0.2, 0.2, 0.1, 0.1
    stability = (max(0.0, 1.0 - abs(vx_new)/vx_tol) *
                 max(0.0, 1.0 - abs(vy_new)/vy_tol) *
                 max(0.0, 1.0 - abs(angle_new)/angle_tol) *
                 max(0.0, 1.0 - abs(angvel_new)/angvel_tol))
    stable_contact_reward = w_stable * double_contact * stability

    # ---------- 水平速度惩罚（辅助对准）----------
    w_horiz_penalty = 0.2
    horizontal_penalty = -w_horiz_penalty * (vx_new**2)

    # ---------- 软安全约束（速度/姿态越界）----------
    w_vel = 0.05
    w_att = 0.05
    vx_max, vy_max = 0.5, 0.5
    angle_max, angvel_max = 0.2, 0.2

    excess_vx = max(0.0, abs(vx_new) - vx_max)
    excess_vy = max(0.0, abs(vy_new) - vy_max)
    vel_penalty = -w_vel * (excess_vx**2 + excess_vy**2)

    excess_angle = max(0.0, abs(angle_new) - angle_max)
    excess_angvel = max(0.0, abs(angvel_new) - angvel_max)
    att_penalty = -w_att * (excess_angle**2 + excess_angvel**2)

    # ---------- 总奖励 ----------
    total_reward = (progress + y_progress + alignment +
                    height_near_reward + any_contact_reward +
                    stable_contact_reward +
                    horizontal_penalty + vel_penalty + att_penalty)

    components = {
        'progress': progress,
        'y_progress': y_progress,
        'alignment': alignment,
        'height_near_reward': height_near_reward,
        'any_contact_reward': any_contact_reward,
        'stable_contact_reward': stable_contact_reward,
        'horizontal_penalty': horizontal_penalty,
        'vel_penalty': vel_penalty,
        'att_penalty': att_penalty
    }
    return float(total_reward), components
```