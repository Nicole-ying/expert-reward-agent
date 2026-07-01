# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 计算当前位置到目标(0,0)的距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist  # 正值表示更接近目标
    progress_delta_reward = 10.0 * progress_delta  # 主信号权重较大

    # ========== 稳定约束：stability_penalty（削弱，避免主导） ==========
    # 惩罚速度、姿态角和角速度，鼓励稳定接近
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = abs(next_obs[4])  # 姿态角偏离0的绝对值
    angular_vel_penalty = abs(next_obs[5])  # 角速度绝对值
    # 大幅降低所有惩罚权重，使stability_penalty不再主导progress信号
    stability_penalty = -0.3 * speed - 0.1 * angle_penalty - 0.05 * angular_vel_penalty

    # ========== 连续着陆塑形：landing_shaping（替换稀疏的soft_landing_proxy） ==========
    # 使用连续函数奖励接近目标+低速+稳定姿态的组合，不再依赖接触标志
    near_target = max(0.0, 1.0 - next_dist / 0.5)  # 距离越近值越大，0.5外为0
    low_speed = max(0.0, 1.0 - speed / 0.5)  # 速度越慢值越大，0.5外为0
    stable_angle = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)  # 角度越小值越大，0.3外为0
    landing_shaping = 3.0 * near_target * low_speed * stable_angle  # 连续乘积，范围[0,3]

    # ========== 动作代价：energy_penalty（小权重） ==========
    # 使用引擎（action != 0）时给予小惩罚
    energy_penalty = -0.1 if action != 0 else 0.0

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + landing_shaping + energy_penalty

    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping": landing_shaping,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```
