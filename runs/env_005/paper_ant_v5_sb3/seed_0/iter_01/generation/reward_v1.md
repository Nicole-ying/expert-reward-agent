# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    body_z = next_obs[0]
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]          # not used but available
    body_x_vel = next_obs[13]
    # body_y_vel and body_z_vel available but not used in v1

    # 1. Forward progress: reward positive forward velocity
    w_forward = 1.0
    forward = w_forward * body_x_vel    # negative when moving backward (natural penalty)

    # 2. Upright orientation: penalty for deviation from perfectly upright
    # body_up_z = 1 - 2*(quat_x^2 + quat_y^2)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_error = 1.0 - body_up_z
    w_upright = 10.0
    upright_pen = -w_upright * upright_error**2

    # 3. Healthy body height: hinge penalty only when approaching termination bounds
    # safe zone (0.25, 0.95), margins before actual termination at [0.2, 1.0]
    w_height = 5.0
    height_pen_low = max(0.0, 0.25 - body_z)
    height_pen_high = max(0.0, body_z - 0.95)
    height_pen = -w_height * (height_pen_low + height_pen_high)

    total_reward = forward + upright_pen + height_pen

    components = {
        "forward": forward,
        "upright": upright_pen,
        "height": height_pen
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family:** locomotion_continuous_control  
- **dynamics_subtype:** multi_legged_body_locomotion (quadruped_3D)  
核心目标是持续向前移动，同时保持身体直立与安全高度。

## selected reward roles
1. **forward_progress** — 主学习信号，驱动机器人产生正向速度。
2. **upright_orientation** — 安全约束，阻止身体过度倾斜导致高度越界/跌倒。
3. **healthy_body_height** — 安全约束，在接近终止边界时施加惩罚，避免过早终止。

## role_to_signal_mapping 与 formula operator 选择

| role | signal(s) used | formula operator | 理由 |
|---|---|---|---|
| forward_progress | `next_obs[13]` (body_x_velocity) | dense_state_signal (线性正奖励：`w * signal`) | 速度是连续、每步有梯度的直接进展指标；负速度自动被惩罚。 |
| upright_orientation | `next_obs[1]..[4]` → 计算 `body_up_z` | quadratic_penalty (`-w * error**2`) | 偏离直立的误差平方惩罚，提供强梯度引导，尤其适合早期防止跌倒。 |
| healthy_body_height | `next_obs[0]` (body_z) | dense_state_signal 的 hinge 形式 (`max(0, threshold - signal)`, `max(0, signal - upper)`) | 只在高度接近安全边界[0.2,1.0]的软边界[0.25,0.95]时生效，避免对安全区内的正常高度施加惩罚，符合专家建议（对明确安全边界的信号用 hinge 而非 quadratic）。 |

## excluded roles 及原因
- **survival**：无 `terminated` 标志可用（info 为空），无法实现每步存活奖励。
- **action_smoothness**：v1 暂不加入，避免早期抑制动作探索。
- **lateral_drift_penalty**、**vertical_oscillation_penalty**：属条件约束，留到后续迭代；v1 优先保证主进展与基本安全。
- **contact_force_reward**、**cumulative_distance_reward**、**explicit_success_bonus**：信号不可用或任务不适合 (见 avoid_roles)。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- 环境未提供显式 success/failure 标志（`explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 false），且 info 字典被禁止使用，因此无法安全地实现终端奖励。

## 留到后续迭代的职责
- **action_smoothness**：可在获得稳定步态后加入小权重动作代价。
- **lateral_drift_penalty**：当侧向漂移明显干扰前进时加入。
- **vertical_oscillation_penalty**：若身体高度周期性大幅波动，可添加 body_z_velocity 惩罚。
- 还可考虑 **soft_health_gate**（把直立或高度作为门控乘到前进奖励上）以及 **joint_condition_proxy**（若需要更软的任务完成引导）。

## 训练后应观察的 failure modes
| failure mode | 可能表现 | 对应干预方向 |
|---|---|---|
| 快速跌倒（高度 < 0.2） | 平均存活步数极低 | 增大 `upright` 和 `height` 权重；考虑 soft_health_gate |
| 原地不动或极低速度 | body_x_velocity 均值接近 0 | 提高 `forward` 权重或对低速度附加凹性激励 |
| 高速但姿态不稳导致终止 | 速度高但存活率低，up_z 波动大 | 提高 `upright` 权重；考虑限制速度上限 |
| 侧向行走或转圈 | body_y_velocity 持续非零 | 后续加入 lateral_drift_penalty |
| 只存活但不前进（奖励 hacking） | 存活但速度接近 0 | 检查权重平衡，必要时提升 `forward` 相对强度或加入速度凸化 |
| 数值崩溃（NaN） | terminated 因 state_value_outside_finite_range | 检查动作/状态是否爆炸；在后续迭代中加入平滑项 |