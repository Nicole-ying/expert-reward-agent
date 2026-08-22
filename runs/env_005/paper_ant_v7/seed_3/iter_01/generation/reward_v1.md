# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------- 从 obs 中提取关键信号 -------------------
    body_z      = obs[0]      # 身体高度
    quat_x      = obs[2]
    quat_y      = obs[3]
    body_x_vel  = obs[13]     # 世界 x 方向前进速度
    body_y_vel  = obs[14]
    body_z_vel  = obs[15]
    roll_vel    = obs[16]
    pitch_vel   = obs[17]
    yaw_vel     = obs[18]

    # ------------------- 派生信号 -------------------
    # body_up_z: 身体在世界坐标系中的“上方向”分量，1 表示完全直立
    # 直接使用环境卡片提供的公式进行计算
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # ------------------- 1. 主学习信号：前进速度 -------------------
    # 使用 linear dense_state_signal，每步都有梯度
    forward_reward = body_x_vel   # 正值鼓励前进，负值惩罚后退

    # ------------------- 2. 姿态约束：直立 -------------------
    # 二次惩罚 body_up_z 偏离 1 的程度（越小越好）
    # 在接近 1 时梯度很小，允许适当的身体摆动；倾斜越大惩罚越强
    upright_error = 1.0 - body_up_z
    upright_penalty = upright_error ** 2
    upright_reward = -upright_penalty

    # ------------------- 3. 高度安全约束：hinge penalty -------------------
    # 只在身体高度接近危险区（<0.3 或 >0.9）时施加惩罚
    # 安全区 (0.3~0.9) 内不惩罚，避免持续抑制正常的运动变化
    height_margin_low  = 0.3
    height_margin_high = 0.9
    below_margin = max(0.0, height_margin_low - body_z)
    above_margin = max(0.0, body_z - height_margin_high)
    height_penalty = below_margin + above_margin
    height_reward = -height_penalty

    # ------------------- 组合 -------------------
    w_forward = 1.0
    w_upright = 0.2
    w_height  = 5.0

    total_reward = (
        w_forward * forward_reward +
        w_upright * upright_reward +
        w_height  * height_reward
    )

    components = {
        'forward_reward': w_forward * forward_reward,
        'upright_reward': w_upright * upright_reward,
        'height_reward':  w_height  * height_reward
    }

    return float(total_reward), components
```

---

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**  
  locomotion_continuous_control / multi_legged_body_locomotion（四足机器人持续前进）。

- **selected reward roles**  
  1. `forward_progress`（主学习信号）  
  2. `upright_posture`（条件约束，始终启用）  
  3. `height_stability`（条件约束，始终启用，使用 hinge）

- **role_to_signal_mapping**  
  - forward_progress → `body_x_velocity` (obs[13])  
  - upright_posture → 由 `quat_x`, `quat_y` 计算得到的 `body_up_z`  
  - height_stability → `body_z` (obs[0])

- **formula operators**  
  - forward_progress: `dense_state_signal` 线性形式  
  - upright_posture: `quadratic_penalty` 形式 `-w * error^2`  
  - height_stability: `dense_state_signal` 的 **hinge** 形式（两侧）

- **excluded roles 及原因**  
  - `energy_efficiency`：v1 阶段保留给后续迭代，避免压制动作探索；  
  - `lateral_vertical_penalty`：属于效率约束，v1 暂不加；  
  - `angular_motion_penalty`：同上；  
  - `survival_bonus`：缺少终止标志且无存活信号，会鼓励原地不动；  
  - `goal_reaching`：无坐标/距离信号；  
  - `contact_smoothness`：无接触力信号。

- **为什么没有 terminal_success_reward / terminal_failure_penalty**  
  `explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，且 `info` 没有任何可用字段，无法实现终端稀疏奖励。

- **留到后续迭代的职责**  
  - 动作幅度惩罚（energy_efficiency）  
  - 横向/垂直速度惩罚（lateral_vertical_penalty）  
  - 角速度惩罚（angular_motion_penalty）  
  - 基于训练进度的动态权重（curriculum_weighting）

- **训练后应重点观察的失败模式**  
  1. **静止不前**：若 `forward_reward` 在 episode 中的总和接近 0，说明 agent 未学会前进，可能需要检查前进速度的激励尺度或试点增大幅值倾斜。  
  2. **姿态崩溃或摔倒**：若终止率高，且 `upright_reward` 在终止前的步中持续为较大负值，说明姿态约束不足，可能需要略微提高 `w_upright` 或改用更大幂次的惩罚。  
  3. **高度越界但未有效惩罚**：若终止前的 `height_reward` 仍然为 0（agent 从安全区一步跳出），说明 hinge 边界过窄，可考虑将 margin 向内收紧（如 low=0.35, high=0.85）。  
  4. **路径依赖的次优步态**：如 agent 学会跳跃前进或大幅侧滑，这些模式在当前 v1 中未直接惩罚，应在后续加入 lateral_vertical 和 angular 惩罚。