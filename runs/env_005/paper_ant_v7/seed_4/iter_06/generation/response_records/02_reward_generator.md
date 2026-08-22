# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract state from next_obs ----------
    body_z = next_obs[0]                     # body height
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]                         # quaternion (not used directly)
    body_x_vel = next_obs[13]                # forward velocity
    body_y_vel = next_obs[14]                # lateral velocity
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]                   # angular velocities

    # ---------- 1. soft health gate: upright factor ----------
    # body_up_z = 1 - 2*(qx^2 + qy^2), ranges from -1 (inverted) to +1 (perfect upright)
    body_up_z = 1.0 - 2.0 * (qx*qx + qy*qy)
    # tilt_error (0 when perfectly upright, up to 2 when inverted)
    tilt_error = 1.0 - body_up_z
    # gate: decays exponentially when tilted, using e^{-lambda * tilt_error}
    lambda_up = 3.0
    upright_gate = 2.718281828 ** (-lambda_up * tilt_error)   # ~1 when upright, ~0 when tilted

    # main progress signal: forward velocity, gated by upright
    gated_forward = body_x_vel * upright_gate

    # ---------- 2. height hinge penalty ----------
    lower_safe = 0.30
    upper_safe = 0.90
    height_penalty = max(0.0, lower_safe - body_z) + max(0.0, body_z - upper_safe)
    # scale to keep penalty comparable but lighter than progress reward
    height_penalty_weighted = -10.0 * height_penalty

    # ---------- 3. lateral velocity penalty ----------
    lateral_penalty = -0.5 * (body_y_vel ** 2)

    # ---------- 4. angular velocity penalty ----------
    ang_penalty = -0.02 * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    # ---------- combine ----------
    w_forward = 2.0
    total_reward = (w_forward * gated_forward
                    + height_penalty_weighted
                    + lateral_penalty
                    + ang_penalty)

    components = {
        "gated_forward": w_forward * gated_forward,
        "height_hinge_penalty": height_penalty_weighted,
        "lateral_penalty": lateral_penalty,
        "angular_velocity_penalty": ang_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task family & dynamics subtype
- **task_family:** locomotion_continuous_control  
- **dynamics_subtype:** multi_legged_body_locomotion (四足机器人)

## selected reward roles
按照环境卡片的 `reward_role_decomposition`，v1 阶段选择了以下职责，合并为四个组件：

| role | type | implementation |
|------|------|----------------|
| forward_velocity_reward (mandatory) & upright_and_orientation_penalty (mandatory) | 主学习信号 + 健康门控 | `gated_forward`：前进速度 × 直立因子（exponential gate），保证只在直立姿态下获得大幅前进奖励，避免“为速度牺牲姿态”的 exploit |
| height_regulation (conditional) | 健康约束 | `height_hinge_penalty`：仅在身体高度越过安全区间 [0.30, 0.90] 时施加惩罚，安全区内无惩罚，避免无故压制正常高度波动 |
| lateral_stability (conditional) | 稳定约束 | `lateral_penalty`：对横向速度平方的轻量惩罚，使前进方向更纯净 |
| rotation_penalty (conditional) | 姿态稳定约束 | `angular_velocity_penalty`：对小角速度的轻量惩罚，抑制剧烈转动和姿态发散 |

## role‑to‑signal mapping

| role | used signals | mapping |
|------|--------------|---------|
| forward_velocity (gated) | `next_obs[13]` (body_x_vel) , `next_obs[2]`,`next_obs[3]` (quat‑x,y → body_up_z) | `gated_forward = body_x_vel * exp(-3 * (1 - body_up_z))` |
| height_hinge | `next_obs[0]` (body_z) | `hinge`: `max(0, 0.3-z) + max(0, z-0.9)`, weighted by -10 |
| lateral_penalty | `next_obs[14]` (body_y_vel) | `quadratic_penalty`: `-0.5 * y_vel^2` |
| angular_penalty | `next_obs[16:19]` (roll,pitch,yaw velocities) | `quadratic_penalty`: `-0.02 * sum(vel^2)` |

所有信号均来自 `next_obs`，不依赖 `info` 或 `original_reward`。

## formula operators 选择理由

- **soft_health_gate** (`upright_gate`)：将直立姿态转化为前进奖励的乘性门控，而不是独立惩罚。这样当身体倾斜时前进奖励被强烈衰减，迫使 agent 必须在直立前提下追求速度，比独立姿态惩罚更直接地防止“快速但因倾斜摔倒”的 failure mode。
- **dense_state_signal – linear**：前进速度的直接线性项（`body_x_vel`）作为核心进度信号，每步有梯度；配合 gate 形成有效学习。
- **dense_state_signal – hinge penalty**：身体高度仅在超过预设安全边界时惩罚。与全局二次惩罚不同，hinge 惩罚让 agent 在安全区内自由寻找步态高度而不被持续惩罚，避免“害怕离开中心高度”的保守行为。
- **quadratic_penalty**：横向速度与角速度使用二次惩罚，因它们不存在绝对“安全区间”，应当被持续但轻量地抑制。

## excluded roles 及原因

- **terminal_success_reward / terminal_failure_penalty**：环境明确无 explicit success/failure flag (`explicit_success_flag_available=false`, `explicit_failure_flag_available=false`)，不可依赖。
- **contact‑based gait rewards**：观测中无脚底接触力信号。
- **absolute position progress**：世界 x 位置被禁止，无法使用。
- **torque / action magnitude penalty**：v1 阶段不加入能耗或动作幅度惩罚，避免过早压制探索（前几个结构出现过低 score 且 agent 可能不愿移动）。效率优化留到后续迭代。
- **joint velocity penalty**：同样暂时搁置，先保证前进与稳定。
- **complex curriculum / dynamic weights**：v1 无需。

## 设计思想与过往尝试的区别

此前尝试（forward_upright_gated + height_gate_value, forward_upright_product, height_survival_penalty 等）均得到极低分数（-9~-500），推测原因包括：
1. 高度约束用 continuous penalty 在整个区间惩罚偏离理想值，导致 agent 不敢采取大动作以维持精确高度，牺牲了前进探索。
2. 多个强独立惩罚（lateral, rotation, torque）叠加压倒了微弱的前进信号，导致 agent “不敢动”或停滞。

本设计的核心变化是：
- 用 **hinge penalty** 替代全时高度惩罚，只在接近危险边界时起效，释放安全区内的高度波动。
- 用 **soft_health_gate** 将直立需求融合进前进信号，避免独立直立惩罚与前进目标竞争。
- 减轻且减少辅助惩罚的数量与强度（去掉 torque, joint velocity 惩罚，轻量保留 lateral 和 angular velocity 惩罚）。
- 通过 gate 使得直立不再是一个需平衡的独立项，而是获得前进奖励的“入场券”。

## 后续迭代预留

- **动作平滑/节能**：在前进步态稳定后引入 action magnitude 或 joint velocity 惩罚，压制高频抖动和能耗。
- **精细步态对称性**：若未来可获得接触信息，可加入基于触地周期的 gait symmetry 奖励。
- **潜在的高度理想值引导**：如果 hinge 区间偏宽导致机器人偏好过低或过高（接近 0.3 或 0.9），可加入软性引导（如温和的正奖励在 0.5‑0.7）。
- **动态课程**：如果发现早期摔倒率极高，可探索用 training_progress 动态调节 gate 灵敏度。

## 训练后应观察的 failure modes

- **stillness / low-speed plateau**：agent 仅保持直立且极小移动，exploting 极小 lateral/angular penalty。若出现，可小幅提高 `w_forward` 或引入前进速度的下限鼓励（如 `max(0, body_x_vel - 0.1)`）。
- **height oscillation near boundaries**：agent 反复在 0.3 和 0.9 之间弹跳以避免惩罚，但步态不稳定。可后续收紧 hinge 区间或加轻微二次惩罚在远边界。
- **tilt‑induced slow death spiral**：开始直立，倾斜后 gate 关闭致奖励骤降，agent 放弃前进，最终摔倒。可考虑 gate 衰减曲线更柔和 (`lambda_up` 调低) 或加入微小的独立直立鼓励。
- **lateral drift**：即使有惩罚，仍可能漂移以保持平衡，推动横向前进捷径。可监控横向速度均值，若持续偏高可加强 lateral 惩罚或添加 yaw 方向奖励。
