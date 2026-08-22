# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v10 – 格式修复版，语义不变
    """

    # 权重
    w_progress = 1.0
    w_speed = 0.3
    w_attitude = 0.1
    w_contact = 0.2
    w_landing = 100.0

    # 当前状态
    x, y = obs[0], obs[1]
    current_dist = (x**2 + y**2) ** 0.5

    # 下一步状态
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    next_dist = (nx**2 + ny**2) ** 0.5
    delta_dist = current_dist - next_dist

    # 组件
    progress_reward = w_progress * delta_dist

    abs_speed = abs(vx) + abs(vy)
    speed_penalty = -w_speed * abs_speed

    attitude_penalty = -w_attitude * (abs(angle) + abs(ang_vel))

    both_legs = left_contact * right_contact       # 0 或 1
    contact_reward = w_contact * both_legs

    landing_condition = (both_legs > 0.5) and (abs_speed < 0.5) and (next_dist < 1.0)
    landing_bonus = w_landing if landing_condition else 0.0

    energy_penalty = 0.0  # 暂时关闭

    total_reward = (
        progress_reward +
        speed_penalty +
        attitude_penalty +
        contact_reward +
        landing_bonus +
        energy_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "speed_penalty": speed_penalty,
        "attitude_penalty": attitude_penalty,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
    }

    return float(total_reward), components
```
