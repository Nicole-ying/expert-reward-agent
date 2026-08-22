# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract key signals from observation
    body_z = obs[0]                # torso height
    qx = obs[2]                    # quaternion x component
    qy = obs[3]                    # quaternion y component
    body_x_vel = obs[13]           # forward (world‑x) velocity

    # ----- uprightness (projection of body z-axis onto world z-axis) -----
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(0.0, min(1.0, body_up_z))   # numerical safety clamp

    # ----- core learning signal: forward velocity modulated by uprightness -----
    w_fwd = 1.0
    forward_reward = w_fwd * body_x_vel * body_up_z

    # ----- safety hinge penalty on torso height -----
    w_h = 0.5
    low_thresh = 0.35
    high_thresh = 0.85
    low_error = max(0.0, low_thresh - body_z)
    high_error = max(0.0, body_z - high_thresh)
    height_penalty = -w_h * (low_error + high_error)

    total_reward = forward_reward + height_penalty

    components = {
        "forward_upright_product": forward_reward,
        "height_survival_penalty": height_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**  
  `locomotion_continuous_control` / `multi_legged_body_locomotion`（四足机器人持续前进）。

- **selected reward roles**  
  1. **主学习信号**：`forward_velocity_reward`（前进速度乘以直立因子，替代纯速度奖励）。  
  2. **健康/安全约束**：`height_survival_hinge`（身体高度软边界惩罚）。  
  3. **稳定约束**：`upright_orientation_gate`（姿态正直性因子，已嵌入主信号乘积中）。

- **role‑to‑signal mapping**  
  - `forward_velocity_reward`  ← `body_x_velocity`（obs[13]）  
  - `upright_orientation_gate` ← `quat_x`, `quat_y`（obs[2], obs[3] → `body_up_z = 1-2(x²+y²)`）  
  - `height_survival_hinge`    ← `body_z`（obs[0]）

- **每个 role 选择的 formula operator**  
  - 前进速度使用 `dense_state_signal` 的**乘积形式**：`body_x_vel * body_up_z`，让前进奖励随身体倾斜程度自然缩放，避免 agent 为了速度而忽略姿态。  
  - 高度约束使用 `dense_state_signal` 的 **hinge penalty**：`-w_h * (max(0, low_thresh - body_z) + max(0, body_z - high_thresh))`，仅在接近终止边界时生效，提供安全区的 soft warning，而不会在正常高度范围内持续惩罚。  
  - 正直性没有独立加减项，而是作为乘积门控融入主信号，等价于 `soft_health_gate` 的简化形式，避免 agent 在能够前进但严重倾斜时仍获得高奖励。

- **excluded roles 及原因**  
  - `terminal_success_reward`、`terminal_failure_penalty`：环境未提供显式成功/失败标志（`explicit_success_flag_available=false`，`info` 为空）。  
  - `lateral_velocity_penalty`、`angular_velocity_penalty`：V1 先建立基础前进+健康行为，侧向与角速度控制留给后续迭代。  
  - `joint_torque_penalty`、`action_smoothness`：无历史动作且能耗优化属于效率次级目标，V1 暂不加入。  
  - `improvement_delta`、`potential_based_shaping`：不存在目标位置度量。

- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**  
  环境不允许使用 `info`，且不存在任何显式终止成功或失败标志，只能从观测构造连续信号。当前设计用身体高度靠近边界时的梯度惩罚防止 agent 早死，而非依赖终局稀疏信号。

- **哪些职责留到后续迭代**  
  - 侧向漂移抑制（`|body_y_velocity|`）  
  - 身体角速度惩罚  
  - 关节扭矩幅度/平滑度惩罚（效率约束）  
  - 更精细的步态或足端接触利用（若将来暴露接触信号）

- **训练后应该观察哪些 failure modes**  
  - **先冲后倒**：agent 在前半段获取高额前进奖励，然后因高度越界或被过度倾斜终止。此时应检查 `height_survival_penalty` 的累积是否太低，以及 `body_up_z` 是否在后期显著下降——若如此，可考虑将高度约束也做成 gate 或加强 hinge 权重。  
  - **原地踏步/站立**：`body_x_vel` 始终接近 0，但高度、姿态均良好，agent 学会“安全地不动”。若发生，需适当降低高度惩罚的保守阈值或增加前进增益。  
  - **大幅摇摆但存活**：身体上下振荡剧烈，但刚好不触发终止，说明 hinge 区间较宽，可能需要收窄阈值或加入角速度轻惩罚。  
  - **reward hacking by oscillating**：agent 利用高度惩罚的零区间反复升降高度来获取前进+高度 combo 而不被罚，此时需要观察高度时序，必要时引入高度变化速率惩罚（但需 `next_obs` 差值或更多历史）。