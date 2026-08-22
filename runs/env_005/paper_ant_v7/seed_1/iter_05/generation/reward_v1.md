# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next_obs after the action
    body_z = next_obs[0]
    qx = next_obs[2]          # quaternion x
    qy = next_obs[3]          # quaternion y
    body_x_vel = next_obs[13] # forward velocity
    body_y_vel = next_obs[14] # lateral velocity

    # 1. Health gate: height factor (safe range [0.3, 0.9])
    if body_z < 0.3:
        height_factor = (body_z - 0.2) / 0.1   # decreases to 0 at z=0.2
    elif body_z > 0.9:
        height_factor = (1.0 - body_z) / 0.1   # decreases to 0 at z=1.0
    else:
        height_factor = 1.0
    height_factor = max(0.0, min(1.0, height_factor))

    # 2. Health gate: upright factor (from body_up_z = 1 - 2*(qx^2 + qy^2))
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    # Map [ -0.2 , 0.5 ] to [0, 1]
    upright_factor = (body_up_z + 0.2) / 0.7
    upright_factor = max(0.0, min(1.0, upright_factor))

    health_gate = height_factor * upright_factor

    # 3. Main learning signal: squared forward velocity, gated by health
    forward_speed = max(0.0, body_x_vel)
    progress_reward = 2.0 * (forward_speed ** 2) * health_gate

    # 4. Light regularization against lateral drift
    lateral_penalty = -0.5 * (body_y_vel ** 2)

    total_reward = progress_reward + lateral_penalty
    components = {
        'forward_progress': progress_reward,
        'lateral_penalty': lateral_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 任务类型与设计背景
- **task_family**: locomotion_continuous_control  
- **dynamics_subtype**: multi_legged_body_locomotion (3D 四足，八关节扭矩控制)  
- **核心目标**: 最大化前向速度，同时保持身体高度在 [0.2, 1.0] 内并维持直立姿态。  
- **无显式 success/failure 标记**，info 不可用，因此不使用 terminal success/failure 奖励。

## 选用的奖励职责与信号映射
根据 `reward_role_decomposition` 的强制性职责，v1 覆盖以下 roles：

| 职责 | 信号 | 公式算子 | 数学形式 |
|------|------|----------|----------|
| forward_velocity_reward (主驱动) | `next_obs[13]` (body_x_velocity) | `dense_state_signal` (凸化) | 平方形式：`2.0 * max(0, v_x)^2`，鼓励高速前移 |
| survival_and_height_reward | `next_obs[0]` (body_z) | `soft_health_gate` (线性衰减门) | 高度因子在 [0.2, 0.3] 和 [0.9, 1.0] 之间线性衰减，核心区域为 1.0 |
| upright_orientation_reward | `next_obs[2:4]` (四元数 x,y) → body_up_z | `soft_health_gate` (线性衰减门) | 直立因子将 body_up_z ∈ [-0.2, 0.5] 映射到 [0, 1] |
| lateral_drift_penalty (条件性) | `next_obs[14]` (body_y_velocity) | `quadratic_penalty` | `-0.5 * v_y^2`，轻量抑制侧向滑动 |

- **高度和直立职责并未以独立惩罚形式出现，而是通过 soft_health_gate 乘在前进奖励上**。  
  这样做是因为：
  - 环境终止的主因是身体高度越界或翻转，单纯加大独立惩罚曾在之前实验中效果不佳（best score 13.55）。
  - soft health gate 在身体状态开始恶化时立即衰减主学习信号，提供稠密梯度，而不是等到越界后才给予已有罚分的稀疏信号。
  - 该 gate 符合 `soft_health_gate` 触发证据：终止率高且前进信号在失败回合仍为正时，需要切断不安全的奖励来源。

## 排除的职责及原因
- **action_regularization**：v1 阶段未加入，避免在探索初期抑制动作多样性，且前期加入动作惩罚的实验表现差（13.55）。后续可迭代加入。
- **angular_rate_penalty**：暂不加入，防止过度抑制动态平衡所需的轻微角速度波动。
- **terminal_success_reward / terminal_failure_penalty**：环境无显式 flag，info 不可用，无法可靠使用。
- **efficiency, smoothness, curriculum** 等默认留到 v2+。

## 关键设计选择
- **凸化前进奖励**：使用 `v_x^2` 而非线性 `v_x`，以防止 agent 满足于低速爬行；系数 2.0 使有效前进时得分量级明显。
- **gate 设计**：
  - 高度安全区间设为 [0.3, 0.9]，比终止边界 [0.2, 1.0] 更收紧，确保在即将违规前梯度已生效。
  - 直立映射允许 body_up_z 低至 -0.2（轻微低头），到 0.5 时 gate 全开，兼顾四足奔跑时的自然俯仰。
- **侧向惩罚**：轻量级（-0.5 * v_y^2），避免侧滑捷径，但不会压制正常侧向平衡调整。

## 预期训练表现与需要观察的 failure modes
- **可能的不安全冲刺**：agent 可能在高 gate 时加速，一旦接近边界 gate 衰减，但惯性可能导致已越界。需观察高度和直立度在 episode 末尾的变化曲线。
- **低速平台**：若凸化奖励仍不足以推动高速，后续可尝试提高系数或引入更陡峭的凸函数（如指数）。
- **侧向爬行**：如果 agent 通过大幅度侧向摆动获得前进（如螃蟹步），可能需要适度增大侧向惩罚或引入偏航角速度惩罚。
- **静止不动**：由于 forward_speed 为 0 时奖励为 0，agent 为了最小化惩罚可能会静止，但 gate 保持 1.0，总奖励为 0 而非负，因此只要有正前进探索就会获得奖励，预计不会导致完全不动。

此 v1 设计放弃了过去“独立惩罚 + 线性 gated”的失败结构，转向以 health-gated 凸化前向奖励为核心，辅以轻量侧向抑制，是一套风格不同的设计假设。