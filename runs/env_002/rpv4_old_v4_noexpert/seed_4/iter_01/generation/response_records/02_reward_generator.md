# Response Record

## 设计说明

- **主学习信号 `forward_progress`**：使用 `next_obs[2]` 的水平速度。采用光滑有界函数 `2 * v / (1 + |v|)`，在速度较小时近似线性奖励，同时避免极端速度数值爆炸。
- **稳定性约束 `upright_stability`**：使用 `next_obs[0]` 躯干倾角和 `next_obs[1]` 躯干角速度，构造光滑有界惩罚，避免摔倒和剧烈摆动。
- **v1 排除项**：
  - 不使用 `original_reward` 或任何未声明的 `info` 字段。
  - 不加入显式 `terminal_success_reward` / `terminal_failure_penalty`，因为环境未提供显式成功/失败旗标。
  - 不加入能量效率惩罚，避免早期抑制探索。
  - 不直接对 LIDAR 做地形高度奖励/惩罚，因为观测中缺少真实地形高度信息。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前向速度，使用光滑有界形式
    horizontal_velocity = next_obs[2]
    forward_progress = 2.0 * horizontal_velocity / (1.0 + abs(horizontal_velocity))

    # 稳定性约束：惩罚躯干倾角与躯干角速度
    torso_angle = next_obs[0]
    torso_angle_sq = torso_angle * torso_angle
    angle_penalty = 1.5 * torso_angle_sq / (1.0 + torso_angle_sq)

    torso_angular_velocity = next_obs[1]
    torso_angular_velocity_sq = torso_angular_velocity * torso_angular_velocity
    angular_velocity_penalty = 0.5 * torso_angular_velocity_sq / (1.0 + torso_angular_velocity_sq)

    upright_stability = -(angle_penalty + angular_velocity_penalty)

    total_reward = forward_progress + upright_stability

    components = {
        "forward_progress": forward_progress,
        "upright_stability": upright_stability,
    }

    return float(total_reward), components
```
