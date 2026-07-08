# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 下一状态的关键信号
    x = next_obs[0]    # 相对目标平台的水平坐标
    y = next_obs[1]    # 相对于平台高度的垂直坐标
    vx = next_obs[2]   # 水平线速度
    vy = next_obs[3]   # 垂直线速度
    angle = next_obs[4] # 机身朝向角度
    omega = next_obs[5] # 角速度

    # 惩罚权重（保持不变）
    w_vx = 0.05
    w_vy = 0.05
    w_angle = 0.05
    w_omega = 0.05

    # 正向接近奖励：使用指数衰减，使得距离0时奖励最大，远离时奖励趋近0
    # 系数 w_target=5 使每维最大奖励5，两维最多10，高于现有惩罚水平
    w_target = 5.0
    target_proximity = w_target * (2.718281828 ** (-abs(x)) + 2.718281828 ** (-abs(y)))

    # 稳定/安全约束：抑制高速运动与机身摆动
    velocity_penalty = -w_vx * (vx ** 2) - w_vy * (vy ** 2)
    orientation_penalty = -w_angle * (angle ** 2)
    angvel_penalty = -w_omega * (omega ** 2)

    total_reward = target_proximity + velocity_penalty + orientation_penalty + angvel_penalty

    components = {
        "target_proximity": target_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "angvel_penalty": angvel_penalty
    }

    return float(total_reward), components
```
