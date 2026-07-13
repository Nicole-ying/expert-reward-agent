# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs (the state after action)
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    av = next_obs[5]
    lc = next_obs[6]
    rc = next_obs[7]

    # Distance to target (landing platform)
    dist = (x**2 + y**2) ** 0.5

    # 1. Goal distance reduction: negative quadratic penalty encourages moving towards zero
    proximity_reward = -0.1 * (x**2 + y**2)

    # 2. Soft landing velocity: velocity penalty gated by proximity
    #    Gate = 1/(1 + k*dist) – strong penalty only when close to platform
    vel_gate = 1.0 / (1.0 + 5.0 * dist)
    velocity_penalty = -0.1 * (vx**2 + vy**2) * vel_gate

    # 3. Upright orientation: penalize tilt and angular velocity
    orientation_penalty = -0.5 * (angle**2) - 0.1 * (av**2)

    # 4. Dual leg contact: sparse bonus when both legs touch the platform
    contact_bonus = 10.0 * lc * rc

    total_reward = proximity_reward + velocity_penalty + orientation_penalty + contact_bonus

    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'contact_bonus': contact_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与动力学子类型
- **task_family:** `navigation_goal_reaching`
- **dynamics_subtype:** `goal_approach_and_soft_contact`（逼近目标、减速、稳定垂直姿态、安全支撑接触）

## 2. 选用的奖励职责（roles）
依据 `reward_role_decomposition`，**必须包含**的三个主职责与一项辅助职责均已覆盖：

| 职责 role_id | 所选 formula operator | 实现方式 |
|---|---|---|
| `goal_distance_reduction` | `dense_state_signal`（二次惩罚） | `proximity_reward = -0.1 * (x² + y²)` |
| `soft_landing_velocity` | `bounded_signal`（倒数门控 × 二次惩罚） | `velocity_penalty = -0.1 * (vx²+vy²) * 1/(1+5*dist)` |
| `upright_orientation` | `quadratic_penalty` | `orientation_penalty = -0.5*angle² - 0.1*av²` |
| `dual_leg_contact` | `logical_and` + `sparse_bonus` | `contact_bonus = 10.0 * lc * rc` |

- `goal_distance_reduction` 作为主学习信号，全程提供稠密梯度。
- `soft_landing_velocity` 仅在与目标距离较近时显著抑制高速，避免过早限制探索。
- `upright_orientation` 防止机体倾倒。
- `dual_leg_contact` 在终局（双腿均接触）时给予一次性正向信号，引导稳定着陆。

## 3. role‑to‑signal 映射验证
- `goal_distance_reduction` 使用 `x_position, y_position`（`next_obs[0:2]`），数学形式为 `dense_state_signal` 中的二次惩罚。
- `soft_landing_velocity` 使用 `x_velocity, y_velocity` 与计算出的 `distance`，通过 `bounded_signal` 的倒数衰减门控实现近距权重提升。
- `upright_orientation` 使用 `body_angle, angular_velocity`，应用 `quadratic_penalty`。
- `dual_leg_contact` 使用 `left_support_contact, right_support_contact`，乘积形式为连续的二值门，等价于环境卡片推荐的 `logical_and`。

## 4. 排除的职责及原因
- `fuel_penalty`（条件职责）：v1 阶段暂不引入，避免早期抑制引擎使用和探索。
- `fast_arrival_bonus`：需要步数计数器，环境未提供，不可用。
- `termination_success_only`：没有显式成功标志（`explicit_success_flag_available=false`），无法实现。
- `safety_collision_penalty`：每步无碰撞信号，且环境不提供 `info` 加以区分，强行使用会引入噪声。

## 5. 未使用 terminal_success_reward / terminal_failure_penalty 的原因
- `explicit_success_flag_available = false`，`explicit_failure_flag_available = false`。
- info 字段为空，无法获取真实的成功或失败原因；直接依赖 terminated 标志会误将沉静或碰撞视为成功/失败，导致奖励信号混淆。
- 因此，奖励设计完全基于每步可观测的连续状态和接触标志，将“成功”分解为近距离 + 零速度 + 竖直 + 双腿接触的组合，并通过 `proximity_reward`、`velocity_penalty`、`orientation_penalty` 和 `contact_bonus` 共同引导。

## 6. 后续迭代可加入的职责
- `fuel_penalty`：待 agent 能稳定到达后，可加入对非零动作的小幅度惩罚，以优化燃料消耗（中后期或 v2）。
- `joint_condition_proxy` 或更精细的软着陆完成度指标：如需在 agent 接近成功但尚未完全满足所有条件时提供额外成形奖励，可构建多个连续因子的乘积或几何平均。

## 7. 训练后应重点观察的 failure modes
| 潜在失败模式 | 检查指标 | 介入建议 |
|---|---|---|
| 高速冲向平台导致弹跳或飞出 | 终局速度平方和很大，双腿接触不稳 | 增大 `velocity_penalty` 系数或降低门控常数 `k` |
| 悬停不降（y 正且缓慢变化） | y 长期正值 | 提高 `proximity_reward` 中 y 的权重，或用平方单独放大 y 项 |
| 姿态失控翻滚 | angle / av 标准差高 | 增大 `orientation_penalty` 系数 |
| 仅单腿接触停滞 | 一腿接触、一腿不接触 | 增强 `contact_bonus` 或引入接近平台时的双侧引导 |
| 过度冲动（early death） | 终局距离还很远 | 后期可考虑加入 `fuel_penalty` 或调整距离奖励权重 |

--- 

以上设计遵循“信号可用性优先”“稠密性”“尺度平衡”与“v1 优先主学习 + 健康约束”的原则，避免过早堆叠效率与复杂门控。