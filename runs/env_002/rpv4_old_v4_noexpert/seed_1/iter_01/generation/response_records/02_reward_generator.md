# Response Record

## Reward v1 设计说明

- 主学习信号：`forward_velocity_reward`，使用 `next_obs[2]` 水平速度，平滑递增，避免硬阈值。
- 健康约束：`upright_stability_penalty`，只惩罚超过死区的躯干角和角速度，保留小幅自然摆动。
- 轻微健康约束：`vertical_oscillation_penalty`，抑制不必要的上下弹跳，但允许正常步态起伏。
- 暂不加入：摔倒惩罚（无显式 failure flag）、能量惩罚（早期会与前进目标冲突）、步态对称性/地形 shaping（条件职责，v1 不加）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度，使用 next_obs 的水平速度
    vx = float(next_obs[2])
    forward_reward = 1.0 - 2.718281828 ** (-max(vx, 0.0) * 1.5)
    forward_reward *= 1.5

    # 稳定性约束：只惩罚过大的躯干倾斜和躯干角速度
    angle_margin = max(0.0, abs(float(next_obs[0])) - 0.15)
    angular_vel_margin = max(0.0, abs(float(next_obs[1])) - 1.0)
    upright_stability_penalty = (
        -2.0 * angle_margin ** 2
        - 0.15 * angular_vel_margin ** 2
    )

    # 上下弹跳约束：轻微惩罚超过死区的垂直速度
    vertical_margin = max(0.0, abs(float(next_obs[3])) - 0.6)
    vertical_oscillation_penalty = -0.3 * vertical_margin ** 2

    total_reward = (
        forward_reward
        + upright_stability_penalty
        + vertical_oscillation_penalty
    )

    components = {
        "forward_velocity_reward": float(forward_reward),
        "upright_stability_penalty": float(upright_stability_penalty),
        "vertical_oscillation_penalty": float(vertical_oscillation_penalty),
    }

    return float(total_reward), components
```
