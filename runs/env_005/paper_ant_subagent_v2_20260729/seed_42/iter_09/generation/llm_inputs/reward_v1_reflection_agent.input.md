# ⚠️ 上一版代码验证失败
错误信息：Reward v1 failed validation: 出现未允许的 obs/next_obs 切片 (record: runs/env_005/paper_ant_subagent_v2_20260729/seed_42/iter_09/generation/validations/reward_v1.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract key signals from next_obs
    body_z = next_obs[0]
    quat_w = next_obs[1]   # not used directly but kept for completeness
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]   # not used directly
    
    forward_vel = next_obs[13]
    lateral_vel = next_obs[14]
    vertical_vel = next_obs[15]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    
    # Derived upright metric: body_up_z
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    
    # --- Reward Components ---
    
    # Main progress: forward velocity (dense, linear)
    forward_reward = 5.0 * forward_vel
    
    # Predictive descent penalty: downward velocity when height is losing margin
    descent_penalty = 0.0
    if body_z < 0.5 and vertical_vel < 0.0:
        descent_penalty = -0.5 * abs(vertical_vel) * (0.5 - body_z)
    
    # Predictive ascent penalty: upward velocity when height is too high
    ascent_penalty = 0.0
    if body_z > 0.8 and vertical_vel > 0.0:
        ascent_penalty = -0.3 * vertical_vel * (body_z - 0.8)
    
    # Lateral drift hinge: penalise excessive sideways motion
    lateral_penalty = -0.2 * max(0.0, abs(lateral_vel) - 0.3)
    
    # Roll / pitch angular velocity hinge (avoids rapid tilting)
    roll_vel_penalty = -0.1 * max(0.0, abs(roll_vel) - 3.0)
    pitch_vel_penalty = -0.1 * max(0.0, abs(pitch_vel) - 3.0)
    
    # Upright tilt penalty: punish large body tilt
    upright_penalty = -0.5 * max(0.0, 0.6 - body_up_z)
    
    # Joint velocity regularisation (very light)
    joint_vels = next_obs[19:27]
    joint_vel_penalty = -0.0005 * sum(v**2 for v in joint_vels)
    
    # Action magnitude regularisation (tiny, analogous to energy cost)
    action_penalty = -0.001 * sum(a**2 for a in action)
    
    total = (forward_reward + descent_penalty + ascent_penalty +
             lateral_penalty + roll_vel_penalty + pitch_vel_penalty +
             upright_penalty + joint_vel_penalty + action_penalty)
    
    components = {
        'forward_reward': forward_reward,
        'descent_penalty': descent_penalty,
        'ascent_penalty': ascent_penalty,
        'lateral_penalty': lateral_penalty,
        'roll_vel_penalty': roll_vel_penalty,
        'pitch_vel_penalty': pitch_vel_penalty,
        'upright_penalty': upright_penalty,
        'joint_vel_penalty': joint_vel_penalty,
        'action_penalty': action_penalty
    }
    
    return float(total), components
```

# reward_v1 设计说明

## 任务与环境画像
- **task_family**: 3D 四足机器人持续前进（locomotion_continuous_control, multi_legged_body_locomotion）
- **控制类型**: 连续扭矩控制（8 关节）
- **核心目标**: 维持身体高度在 (0.2, 1.0)、保持直立、产生持续的前进速度；次要目标为最大化前进速度
- **信号可用性**: 有身体高度、四元数、身体速度、关节速度等密集信号；无显式成功/失败标志，无全局位置，无接触信息

## 设计思路：预测性惩罚代替硬门控

历史实验表明，将前进奖励与健康门控（gate）相乘极易导致奖励塌缩或探索不足，单纯的多项独立惩罚又使总奖励过度负化。本次设计采取“**主奖励 + 预测性轻量惩罚**”策略，核心假设是：

- **主奖励（forward_reward）** 提供直接的向前驱动力，使用 `5.0 * forward_vel`（线性），数值足够大以保证正反馈；
- **预测性惩罚** 在危险动作（下降速度、上升速度、侧向漂移、快速翻滚/俯仰角速度）刚刚出现时给予轻度梯度，防止 agent 滑向不可恢复的失败状态，而不是等身体已经越界再施加过强惩罚；
- **轻量约束**（关节速度、动作幅度）抑制高频振荡，但不压制探索。

这种结构从 iter 4 的唯一正分实验（descent_early_penalty + forward_progress + 少量约束）获得启发，并将其扩展为更全面的预测规则。

## 选择的奖励职责与公式算子映射

| 职责 (role) | 使用信号 | 公式算子 | 权重与阈值 | 说明 |
|---|---|---|---|---|
| forward_velocity_progress | next_obs[13] | dense_state_signal (线性) | `+5.0 * forward_vel` | 核心学习信号，推动机器人持续前进 |
| descent_early_penalty | next_obs[0], [15] | joint_condition_proxy (乘积 hinge) | `-0.5 * \|v_z\| * max(0, 0.5 - h)`（当 h<0.5 且 v_z<0） | 提前惩罚下降速度，避免摔倒前兆无反馈 |
| ascent_early_penalty | next_obs[0], [15] | joint_condition_proxy (乘积 hinge) | `-0.3 * v_z * max(0, h - 0.8)`（当 h>0.8 且 v_z>0） | 防止跳起导致高度越界 |
| lateral_drift_penalty | next_obs[14] | dense_state_signal (hinge) | `-0.2 * max(0, \|v_y\| - 0.3)` | 抑制侧向漂移 |
| roll_pitch_angular_penalty | next_obs[16], [17] | dense_state_signal (hinge) | `-0.1 * max(0, \|\omega_roll\| - 3.0)` 等同 | 防止快速翻滚/俯仰，避免失稳 |
| upright_penalty | next_obs[2:3] → body_up_z | dense_state_signal (hinge) | `-0.5 * max(0, 0.6 - body_up_z)` | 倾斜过大时给予温和惩罚 |
| joint_velocity_regularization | next_obs[19:27] | quadratic_penalty (L2) | `-0.0005 * Σ v_joint²` | 极轻量抑制关节振荡 |
| action_magnitude | action[0:7] | quadratic_penalty (L2) | `-0.001 * Σ a²` | 微小的动作幅度成本 |

## 排除的角色及原因

- **healthy_height_gate**（原 mandatory）：多次实验证明门控会导致奖励塌缩；本设计通过预测性惩罚间接实现保护，效果更平滑。
- **upright_penalty 作为独立 gate**：已替换为轻量 hinge 惩罚，避免关闭前进奖励。
- **energy_efficiency**：无直接能量信号，且任务未要求省能。
- **target_reaching / distance_to_goal**：无全局 x 坐标。
- **foot_contact_reward**：

# 1. Search objective
- target_score: 3000.000000
- current_score: -415.211675
- gap_to_target: 3415.211675
- target_achievement_ratio: -13.840%

# 2. 上一轮奖励函数代码（该轮得分: -415.211675）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    forward_vel = next_obs[13]   # body x velocity
    lateral_vel = next_obs[14]   # body y velocity
    roll_vel    = next_obs[16]   # body roll velocity
    pitch_vel   = next_obs[17]   # body pitch velocity
    body_z      = next_obs[0]    # body height

    # 计算身体直立度：body_up_z = 1 - 2*(qx^2 + qy^2)
    qx = next_obs[2]
    qy = next_obs[3]
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)  # 范围[-1,1]，1为完全直立

    # 直立性门控：当 body_up_z 低于 0.3 时门控归零，线性过渡到 1.0 时门控为 1
    uprightness_gate = max(0.0, (body_up_z - 0.3) / 0.7)

    # 主前进奖励（引入直立性门控）
    forward_reward = 2.0 * forward_vel * uprightness_gate

    # 侧向漂移惩罚 (hinge, 容忍 |lateral_vel| <= 0.2)
    lateral_penalty = -0.5 * max(0.0, abs(lateral_vel) - 0.2)

    # 滚转/俯仰角速度惩罚 (hinge, 容忍 |roll|,|pitch| <= 0.5)
    roll_penalty  = -0.2 * max(0.0, abs(roll_vel) - 0.5)
    pitch_penalty = -0.2 * max(0.0, abs(pitch_vel) - 0.5)

    # 身体高度边界惩罚 (hinge: 安全区域 [0.3, 0.9])
    height_low_penalty  = -1.0 * max(0.0, 0.3 - body_z)
    height_high_penalty = -1.0 * max(0.0, body_z - 0.9)

    total_reward = (forward_reward +
                    lateral_penalty + roll_penalty + pitch_penalty +
                    height_low_penalty + height_high_penalty)

    components = {
        'forward_reward': forward_reward,
        'lateral_penalty': lateral_penalty,
        'roll_penalty': roll_penalty,
        'pitch_penalty': pitch_penalty,
        'height_low_penalty': height_low_penalty,
        'height_high_penalty': height_high_penalty,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-415.211675, len=299.050000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-1702.611675, 18.873805]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 74.738484 | 38.8% | 72.1% | 75.5% |
| lateral_penalty | -17.182856 | -8.9% | 8.9% | 21.6% |
| roll_penalty | -15.075047 | -7.8% | 7.8% | 21.1% |
| pitch_penalty | -12.114540 | -6.3% | 6.3% | 19.1% |
| height_low_penalty | -8.964256 | -4.7% | 4.7% | 74.7% |
| height_high_penalty | -0.301906 | -0.2% | 0.2% | 1.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
机器人需要沿着世界坐标系 x 轴（前进方向）持续、稳定地移动。主要目标是维持稳定的前进行走，而非仅仅保持直立。在实现稳定行走后，更快的向前速度应获得更高评价，但不能以牺牲稳定性为代价。身体高度必须保持在健康范围内，过低（摔倒）或过高（弹跳/飞起）都会导致回合终止。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: 连续浮点（推测 float32/64）
- obs[0]: body_z，身体垂直位置（高度），reward_usable: true
- obs[1]: quat_w，身体朝向四元数实部，reward_usable: true（需转换为 uprightness）
- obs[2]: quat_x，身体朝向四元数第一虚部，reward_usable: true（需转换为 uprightness）
- obs[3]: quat_y，身体朝向四元数第二虚部，reward_usable: true（需转换为 uprightness）
- obs[4]: quat_z，身体朝向四元数第三虚部，reward_usable: true（需转换为 uprightness）
- obs[5]: joint_1_angle，第一髋关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[6]: joint_2_angle，第一踝关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[7]: joint_3_angle，第二髋关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[8]: joint_4_angle，第二踝关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[9]: joint_5_angle，第三髋关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[10]: joint_6_angle，第三踝关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[11]: joint_7_angle，第四髋关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[12]: joint_8_angle，第四踝关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[13]: body_x_velocity，身体世界 x 轴速度（前进速度），reward_usable: true（主信号核心）
- obs[14]: body_y_velocity，身体世界 y 轴速度（侧向速度），reward_usable: true（抑制侧向漂移）
- obs[15]: body_z_velocity，身体垂直速度，reward_usable: true（监测坠落/弹跳）
- obs[16]: body_roll_velocity，翻滚角速度，reward_usable: true（稳定性指标）
- obs[17]: body_pitch_velocity，俯仰角速度，reward_usable: true（稳定性指标）
- obs[18]: body_yaw_velocity，偏航角速度，reward_usable: true（前进方向稳定性）
- obs[19]: joint_1_velocity，第一髋关节角速度，reward_usable: true
- obs[20]: joint_2_velocity，第一踝关节角速度，reward_usable: true
- obs[21]: joint_3_velocity，第二髋关节角速度，reward_usable: true
- obs[22]: joint_4_velocity，第二踝关节角速度，reward_usable: true
- obs[23]: joint_5_velocity，第三髋关节角速度，reward_usable: true
- obs[24]: joint_6_velocity，第三踝关节角速度，reward_usable: true
- obs[25]: joint_7_velocity，第四髋关节角速度，reward_usable: true
- obs[26]: joint_8_velocity，第四踝关节角速度，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: [8]
- n: 8 维连续动作
- action[0]: hip_1_torque，第一个髋关节扭矩，范围 [-1.0, 1.0]
- action[1]: ankle_1_torque，第一个踝关节扭矩，范围 [-1.0, 1.0]
- action[2]: hip_2_torque，第二个髋关节扭矩，范围 [-1.0, 1.0]
- action[3]: ankle_2_torque，第二个踝关节扭矩，范围 [-1.0, 1.0]
- action[4]: hip_3_torque，第三个髋关节扭矩，范围 [-1.0, 1.0]
- action[5]: ankle_3_torque，第三个踝关节扭矩，范围 [-1.0, 1.0]
- action[6]: hip_4_torque，第四个髋关节扭矩，范围 [-1.0, 1.0]
- action[7]: ankle_4_torque，第四个踝关节扭矩，范围 [-1.0, 1.0]

> 动作均为关节扭矩（torque），值为归一化到 [-1, 1] 后的控制信号。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明显明确成功终止。环境可能一直运行直到 truncation 或失败终止。
- failure-like termination:
  - body_height_outside_healthy_range: 身体高度超出 [0.2, 1.0]，过低（摔倒/坍塌）或过高（弹跳/飞起）。
  - state_value_outside_finite_range: 任意状态值变为 NaN 或无穷大（仿真崩溃/数值溢出）。
- ambiguous termination:
  - 无显式成功信号。如果 episode 在终止前未触发 failure，可能是在稳定行走中因步数上限 truncated。
- truncation: time_limit_reached（步数上限截断）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false (failure 仅通过 terminated 标志间接体现，但不能区分失败类型)
- allowed_info_fields: []（无可用 info 字段）
- forbidden_or_uncertain_info_fields:
  - reward_forward, reward_ctrl, reward_contact, reward_survive（官方奖励分量，禁止使用）
  - x_position, y_position, distance_from_origin（绝对位置信息，禁止使用）

> 虽然没有显式 flag 可用，但可通过观测信号间接推断失败：
> - 摔倒/坍塌: body_z（obs[0]）急剧下降并接近或低于 0.2，或前一帧 body_z 明显偏高但被截断前突然更低。
> - 弹跳/飞起: body_z 突然超过 1.0 或 body_z_velocity 为正异常值。
> - 方向失控/倒地: body_up_z（由四元数计算）偏离 1.0 很严重，且 body_roll/pitch_velocity 急剧变化。
> - 仿真崩溃: 观测值出现 NaN/Inf 在终止前可能无法读取，但可提前检测。

## 7. 可用于奖励函数的信号
- position:
  - body_z (obs[0]): 身体高度，可判断是否在健康范围 [0.2, 1.0]
  - joint_angles (obs[5:13]): 关节角度，可用于动作平滑性或姿态评估
- velocity:
  - body_x_velocity (obs[13]): 前进速度（主信号核心）
  - body_y_velocity (obs[14]): 侧向速度（应抑制漂移）
  - body_z_velocity (obs[15]): 垂直速度（可检测弹跳/坠落趋势）
  - body_roll_velocity (obs[16]): 翻滚角速度（稳定性）
  - body_pitch_velocity (obs[17]): 俯仰角速度（稳定性）
  - body_yaw_velocity (obs[18]): 偏航角速度（方向稳定性）
  - joint_velocities (obs[19:27]): 关节角速度
- orientation:
  - body orientation quaternion (obs[1:5]): 可计算 body_up_z = 1 - 2*(quat_x² + quat_y²)，范围[-1, 1]，1 表示完全直立。可用于评估是否倾斜过度。
- contact: 无接触力或接触标志信号可用
- action/engine:
  - action (8D): 关节扭矩控制信号，可用于动作效率（能耗）奖励
- other:
  - derived_possible: 失败推断信号（见 5.2），通过 body_z、四元数、速度异常变化间接推测终止原因。