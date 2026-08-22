# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- extract relevant observations ----
    body_z = obs[0]
    # quaternion components
    quat_x = obs[2]
    quat_y = obs[3]
    # forward (world x) velocity
    body_x_vel = obs[13]

    # ---- compute health indicators ----
    # target body height (midpoint of healthy range)
    target_height = 0.6
    height_error = body_z - target_height

    # body up vector z-component in world frame (derived from quaternion)
    # body_up = [2*(qx*qz + qw*qy), 2*(qy*qz - qw*qx), 1 - 2*(qx^2 + qy^2)]
    body_up_z = 1.0 - 2.0 * (quat_x * quat_x + quat_y * quat_y)

    # tilt error: 0 when perfectly upright, >0 when tilting
    tilt_error = 1.0 - body_up_z

    # combine height and tilt errors into a single posture error
    posture_error = (height_error * height_error + tilt_error * tilt_error) ** 0.5

    # exponential health gate: 1 when posture_error = 0, decays towards 0 as error grows
    k_gate = 5.0
    health_gate = 2.718281828 ** (-k_gate * posture_error)

    # ---- forward progress signal ----
    # only reward moving forward (positive x velocity)  
    forward_speed = body_x_vel if body_x_vel > 0.0 else 0.0
    # bounded mapping to avoid unbounded credit for extremely high speeds
    forward_bounded = forward_speed / (1.0 + forward_speed)

    # ---- total reward ----
    # multiplicative combination: agent must maintain good posture to earn forward progress credit
    w_forward = 8.0
    total_reward = w_forward * forward_bounded * health_gate

    # components dict for logging (total_reward itself is the only additive component)
    components = {
        'progress': total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**：`locomotion_continuous_control`，四足机器人通过 8 维扭矩控制实现稳定前向行走，同时保持身体高度与姿态。
- **selected reward roles**：
  - **主学习信号**：利用 `body_x_velocity` 的正向部分，经饱和映射（`x/(1+|x|)`）生成逐步前进奖励，避免极端速度导致奖励爆炸。
  - **健康/姿态门控**：基于身体高度（目标 0.6 m）和身体直立程度（`body_up_z`）构造指数衰减门 `health_gate`，与前向奖励相乘。这使得机器人必须同时满足高度和姿态要求才能获得前进分数，避免了独立惩罚项对探索的早期压制。
- **role_to_signal_mapping**：
  - 前向速度：`obs[13]` → `forward_speed` → `forward_bounded`
  - 高度偏差：`obs[0]` → 对目标高度 0.6 的差值
  - 身体直立：`obs[1:5]`（四元数）计算 `body_up_z` → 倾角误差
- **每个 role 选择的 formula operator**：
  - `forward_bounded`：`bounded_signal`（饱和映射）
  - `health_gate`：`soft_health_gate`（指数衰减形 `exp(-k * posture_error)`），用乘积方式调制主线奖励，而不是加法惩罚。
- **excluded roles 及原因**：
  - `terminal_success_reward`：环境无显式成功标志（`explicit_success_flag_available=false`），且 `info` 不可用，无法实现。
  - `terminal_failure_penalty`：无显式失败标志，且 `info` 被禁用。
  - `efficiency/action penalty`：按 v1 预算留在后续迭代。
  - `improvement_delta` 等未采用，因当前设计已提供稠密梯度；侧滑和角速度约束通过 posture error 间接涵盖，不再单独增加约束组件。
- **为什么没有使用 terminal 奖励**：环境终止由身体高度出界或状态发散触发，但奖励函数内部无法获取终止信号（`info` 为空），因此必须完全依赖观测值构建稠密引导。
- **哪些职责留到后续迭代**：侧向漂移的直接约束、角速度平滑项、能耗/动作幅度代价、多阶段课程学习等，将在后续版本中引入。
- **训练后应观察的 failure modes**：
  - 机器人可能学会“慢速踱步”以获得高门控值但前进微弱；检查前进速度分布。
  - `health_gate` 可能抑制早期探索，观察训练前期身体是否过于僵硬。
  - 若 `k_gate` 过大或 `w_forward` 太小，可能导致总奖励过低，代理陷入局部不动策略；需在下一迭代中调参。
  - 注意代理是否利用饱和前向奖励的特性，仅以中等速度行走而不尝试更高效步态。
