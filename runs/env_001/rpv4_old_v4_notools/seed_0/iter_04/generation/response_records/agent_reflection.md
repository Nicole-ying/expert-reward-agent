# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: 将 contact_reward 从极度稀疏的落地瞬间奖励改写为渐进式高度相关奖励，
    使 agent 在整个下降过程中都能获得连续的正向引导。
    progress / orientation / vel_damping 保持与 v3 相同的公式，仅微调权重以配合新的 contact 幅度。
    """
    # 提取信号
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward（保持 v3 权重）
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr
    w_progress = 0.5
    progress_reward = w_progress * progress

    # 2. Orientation penalty（保持 v3）
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. 渐进式接触/高度奖励（v4 核心修改）
    alt = y_curr
    # 高度因子：越接近地面奖励越大，高度 > 2.0 时几乎为 0
    alt_factor = 2.718281828 ** (-alt * 1.5)
    # 水平偏离中心惩罚（目标 x=0）
    lateral_penalty = 2.718281828 ** (- (x_curr ** 2) / 0.5)
    # 速度因子：接近地面时速度越小越好
    speed = (vx ** 2 + vy ** 2) ** 0.5
    vel_factor = 2.718281828 ** (-speed * 2.0)
    # 姿态因子：保持竖直
    attitude_factor = 2.718281828 ** (- (angle ** 2) / 0.2)
    # 脚的接触直接加分（但权重不高，仅作为接触确认）
    contact_bonus = 10.0 * (left_contact + right_contact)
    # 组装
    w_contact = 30.0  # 降低权重，避免一次接触支配总额
    contact_reward = w_contact * (alt_factor * lateral_penalty * vel_factor * attitude_factor + 0.1 * contact_bonus / 10.0)

    # 4. 接近地面速度惩罚（保持 v3 结构，微调系数）
    alt_thresh = 1.0
    gate_alt = max(0.0, 1.0 - alt / alt_thresh)
    w_vel_damp = 1.0
    vel_damping = -w_vel_damp * gate_alt * (abs(vy) + 0.3 * abs(vx))

    total_reward = progress_reward + orientation_penalty + contact_reward + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward,
        "vel_damping": vel_damping
    }
    return float(total_reward), components
```
