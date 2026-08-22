# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆解
    body_z = obs[0]
    quat_w, quat_x, quat_y, quat_z = obs[1], obs[2], obs[3], obs[4]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]    # 可能用于横向约束，但v1先不加入
    body_yaw_velocity = obs[18]  # 可能用于偏航约束，但v1先不加入

    next_body_z = next_obs[0]

    # 计算 body_up_z：躯干对世界 z 轴的投影
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 1. 主学习信号：前向速度奖励（凸化形式） ----------
    # 使用平方形式，在 signal 较大时提供更强梯度，推动 agent 从低速平台突破
    forward_speed_reward = 0.5 * (body_x_velocity ** 2)

    # ---------- 2. 健康/安全约束 ----------
    # 2a. 身体高度合页惩罚
    # 安全区域 [0.25, 0.95]，在边界附近开始软惩罚；阈值设在终止边界的 60-80%
    z_lower = 0.25
    z_upper = 0.95
    z_lower_hinge = max(0.0, z_lower - body_z)       # body_z 低于 0.25 时 >0
    z_upper_hinge = max(0.0, body_z - z_upper)       # body_z 高于 0.95 时 >0
    height_penalty = -0.3 * (z_lower_hinge**2 + z_upper_hinge**2)

    # 2b. 直立姿态惩罚（二次形式）
    # body_up_z 的理想范围为 0.85~1.0，中心 1.0
    # 使用 (1 - body_up_z)**2；当 body_up_z 在 0.8 左右时 penalty 约 0.04，在可接受范围
    upright_penalty = -0.05 * ((1.0 - body_up_z) ** 2)

    # ---------- 总奖励 ----------
    total_reward = forward_speed_reward + height_penalty + upright_penalty

    components = {
        "forward_speed_reward": forward_speed_reward,
        "height_penalty": height_penalty,
        "upright_penalty": upright_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- `locomotion_continuous_control` / `multi_legged_body_locomotion`
- 四足机器人通过 8 个连续扭矩关节驱动前进。

## selected reward roles
1. **forward_velocity**（主学习信号）：鼓励持续向前运动，是任务的核心目标。采用凸化形式（平方）以在速度偏小时提供更强梯度，帮助突破低速平台。
2. **height_survival**（安全约束）：维持身体高度在健康范围。使用合页惩罚，只在接近终止边界时生效，避免在安全区域内持续惩罚正常波动。下界阈值设在 0.25 m，上界 0.95 m，略大于终止边界的 0.2 和 1.0，提供缓冲。
3. **upright_orientation**（姿态约束）：保持躯干大致直立，防止翻滚。使用二次惩罚于 `1 - body_up_z`，允许轻微前倾，不对正常步态产生过度抑制。

## role_to_signal_mapping
| role | signal | index / derivation |
|------|--------|--------------------|
| forward_velocity | `body_x_velocity` | obs[13] |
| height_survival | `body_z` | obs[0] |
| upright_orientation | `body_up_z = 1 - 2*(qx² + qy²)` | derived from obs[1:5] |

## formula operator selection
- **forward_velocity → dense_state_signal（凸化）**：平方形式 `0.5 * v²`。触发证据来自历史尝试中 score 停滞在低水平，agent 可能陷入低速稳态，凸化有助于打破平台。
- **height_survival → dense_state_signal（hinge）**：合页惩罚 `-0.3 * (hinge²)`。历史尝试中常出现“全时惩罚仍高 terminated”，hinge 只在越界（或接近越界）时生效，避免安全区内无效惩罚。
- **upright_orientation → quadratic_penalty**：`-0.05 * (1-up_z)²`。该约束为轻量级，不应压制主前进信号。

## excluded roles 及原因
- **lateral_and_yaw_penalty**：当前尚未观察横向漂移是否为主要失败模式；v1 默认不加，待训练后分析失效模式再决定是否引入。
- **action_smoothness / torque_cost**：v1 阶段避免引入效率项，能耗和抖动优化留到后续迭代。
- **global_progress_reward / contact_force_penalty**：所需信号（x/y 位置、接触力）不可用。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- `explicit_success_flag_available=false`，`explicit_failure_flag_available=false`。
- 无 `info` 字段提供成功/失败标志，无法实现。
- 替代方案：通过高度合页惩罚在越界前提供梯度，间接指导存活。

## 哪些职责留到后续迭代
- 横向/偏航抑制、动作平滑、扭矩效率、动态课程权重等。

## 训练后应该观察哪些 failure modes
- **低速平台**（前向速度停滞在较低正值）：可能需要进一步加大凸化指数或调整系数。
- **高度边界震荡**（身体反复触碰合页边界）：可能需要微调 hinge 阈值或增加缓冲宽度。
- **前倾摔倒**（body_z 快速下降且 upright_penalty 增大但仍终止）：可能需要加强高度合页惩罚的上界部分或考虑将高度作为 soft gate 乘到前向奖励上。
- **侧翻/偏航旋转**（body_up_z 骤降且 body_yaw_velocity 异常）：后续需加入 lateral_and_yaw_penalty。
