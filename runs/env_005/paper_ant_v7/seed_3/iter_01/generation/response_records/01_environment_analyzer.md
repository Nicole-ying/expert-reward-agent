# Response Record

# 匿名环境理解卡片

## 1. 任务目标
这是一个 3D 四足机器人连续控制任务。机器人拥有四条腿、八个力矩控制关节，需要在保持身体直立且高度处于健康范围的前提下，尽可能稳定地向前行走或奔跑。主要目标是持续、快速的**前进运动**，而非仅仅保持平衡或存活。次要目标可包括维持身体姿态稳定、动作平滑、能量高效，但这些都服务于前进这一核心目标。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务的核心驱动力是让机器人在连续控制下产生前进运动，没有明确的终点或抓取目标；附属的姿态、高度约束属于健康条件，不是平行主目标，因此符合 locomotion_continuous_control 类型。

dynamics_subtype: multi_legged_body_locomotion  
说明：多足身体在地面上的持续步态推进，具有高维身体、关节力矩驱动、接触不直接可测的特点。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float32（推断）
- 维度含义表（索引 0~26）

| obs index | name | meaning | reward_usable |
|---|---|---|---|
| 0 | body_z | 机器人主体的垂直高度 | true |
| 1 | quat_w | 身体方向四元数实部 | true |
| 2 | quat_x | 身体方向四元数虚部 x | true |
| 3 | quat_y | 身体方向四元数虚部 y | true |
| 4 | quat_z | 身体方向四元数虚部 z | true |
| 5 | joint_1_angle | 第 1 髋关节角度 | true |
| 6 | joint_2_angle | 第 1 踝关节角度 | true |
| 7 | joint_3_angle | 第 2 髋关节角度 | true |
| 8 | joint_4_angle | 第 2 踝关节角度 | true |
| 9 | joint_5_angle | 第 3 髋关节角度 | true |
| 10 | joint_6_angle | 第 3 踝关节角度 | true |
| 11 | joint_7_angle | 第 4 髋关节角度 | true |
| 12 | joint_8_angle | 第 4 踝关节角度 | true |
| 13 | body_x_velocity | 世界 x 方向前进速度 | true |
| 14 | body_y_velocity | 世界 y 方向横向速度 | true |
| 15 | body_z_velocity | 垂直速度 | true |
| 16 | body_roll_velocity | 滚转角速度 | true |
| 17 | body_pitch_velocity | 俯仰角速度 | true |
| 18 | body_yaw_velocity | 偏航角速度 | true |
| 19 | joint_1_velocity | 第 1 髋关节角速度 | true |
| 20 | joint_2_velocity | 第 1 踝关节角速度 | true |
| 21 | joint_3_velocity | 第 2 髋关节角速度 | true |
| 22 | joint_4_velocity | 第 2 踝关节角速度 | true |
| 23 | joint_5_velocity | 第 3 髋关节角速度 | true |
| 24 | joint_6_velocity | 第 3 踝关节角速度 | true |
| 25 | joint_7_velocity | 第 4 髋关节角速度 | true |
| 26 | joint_8_velocity | 第 4 踝关节角速度 | true |

额外可用派生：  
- body_up_z = 1 - 2*(quat_x² + quat_y²)，范围 [-1,1]，1 表示完全直立，可用于姿态奖励。
- 所有关节角度、速度可用于动作平滑或关节姿态惩罚。

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- continuous: true
- bounds: [-1.0, 1.0] per joint（对应标准化力矩）

| action dim | name | meaning |
|---|---|---|
| 0 | hip_1_torque | 第 1 髋关节力矩 |
| 1 | ankle_1_torque | 第 1 踝关节力矩 |
| 2 | hip_2_torque | 第 2 髋关节力矩 |
| 3 | ankle_2_torque | 第 2 踝关节力矩 |
| 4 | hip_3_torque | 第 3 髋关节力矩 |
| 5 | ankle_3_torque | 第 3 踝关节力矩 |
| 6 | hip_4_torque | 第 4 髋关节力矩 |
| 7 | ankle_4_torque | 第 4 踝关节力矩 |

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: 无显式成功终止标志。
- **failure-like termination**:  
  - body_height_outside_healthy_range: 主体垂直高度 ≤ 0.2 或 ≥ 1.0 时立即终止（可分别视为摔倒或腾空失控）。  
  - state_value_outside_finite_range: 任意状态值变为 NaN 或无穷大时终止（数值崩溃）。
- **ambiguous termination**:  
  - truncated = time_limit_reached，仅代表时间耗尽，不能直接诠释为成功或失败。
- **truncation**: 由时间限制触发。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false  
  （虽然终止原因可推断为高度越界或数值异常，但在 compute_reward 接口中无法获取 terminated 标志，只能通过 next_obs 的有限信息间接判断。）
- allowed_info_fields: []（本環境不允许在 reward 中使用任何 info 字段）
- forbidden_or_uncertain_info_fields:  
  reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等均不可用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（27维）
- action（8维）
- next_obs（27维）
- training_progress（仅在 prompt 明确允许时）
- info 中明确允许的字段（本环境 **无**）

禁止使用：
- original_reward（被强制遮蔽）
- official_reward
- 未声明的 info 字段（全部 info 字段均不可用）
- 未声明的 obs/next_obs 切片（但可使用派生量，如 body_up_z）
- 任何来自环境内部的 x/y 世界坐标、接触力等

## 7. 可用于奖励函数的信号
- **position**:  
  - body_z (高度，可做高度保持奖励)  
  - 四元数 → 直立度 body_up_z  
  - 关节角度（可用于姿态正则化）
- **velocity**:  
  - body_x_velocity (世界 x 方向前进速度，核心前进信号)  
  - body_y_velocity, body_z_velocity（横向、垂直速度，可用于惩罚非前进方向运动）
  - 身体角速度 (roll, pitch, yaw) 及关节角速度（可用于平稳性惩罚）
- **orientation**:  
  - 通过四元数计算直立即时状态
- **contact**: 无（此环境无接触力信息）
- **action/engine**:  
  - 8 个关节力矩（可用于动作幅度惩罚、平滑性惩罚）
- **other**:  
  - next_obs 与 obs 的差分可用于瞬时变化量。

## 8. 不确定或不可用的信号
- 世界坐标 x, y（不可用）
- 前进距离或基准位置（不可用）
- 接触力 / 地面反作用力（不可用）
- 显式存活奖励、控制代价等原环境奖励分量（不可用）
- terminated 标志（在 reward 函数签名中未提供，亦不可从 info 获取）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped
  actuator_type: torque_controlled_joints (8 DoF)
  contact_structure: four legs, no direct contact force observable
primary_objectives:
  - maximize forward velocity (body_x_velocity) over episode
secondary_objectives:
  - maintain upright posture (body_up_z close to 1)
  - keep body height stable within healthy range (avoid approaching 0.2 or 1.0)
  - minimize lateral/vertical velocity and excessive angular velocity
  - reduce energy consumption via moderate joint torques
main_failure_risks:
  - falling over (body height drops below 0.2)
  - launching or unstable jumping (body height exceeds 1.0)
  - instability causing NaN/inf states
  - learning to stand still without forward progress
  - excessive joint oscillations or jerky motion
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- **role_id**: forward_progress  
  purpose: 鼓励机器人持续沿 x 方向前进，是任务核心驱动源。  
  why_required: 任务描述明确要求“walk or run forward”，且无其他到达性目标。  
  usable_signals: [body_x_velocity (obs[13])]  
  risks:  
    - 若奖励过强可能导致机器人为求高速而姿态失控；  
    - 需配合姿态/高度约束避免策略学习到跳跃前进的劣质行为。

### 10.2 条件职责 conditional_roles
- **role_id**: upright_posture  
  condition_to_use: 当前进奖励已存在前提下，建议始终施加，以过滤姿态崩溃的策略。  
  usable_signals: [body_up_z (由 quat_x, quat_y 计算)]  
  risks: 过度强调直立可能降低步态的多样性，抑制快速奔跑时的自然身体摆动。

- **role_id**: height_stability  
  condition_to_use: 建议始终施加，但可设为弱奖励，仅当高度接近阈值（如 <0.3 或 >0.9）时才显著惩罚。  
  usable_signals: [body_z (obs[0])]  
  risks: 与前进奖励可能冲突，若惩罚过重会阻碍探索；最好设计为在安全范围外有边际增加。

- **role_id**: energy_efficiency  
  condition_to_use: 在训练早期可关闭或权重较低，后期或需要稳定表现时加入，以减少关节力矩消耗。  
  usable_signals: [action (8维扭矩)]  
  risks: 过度惩罚动作幅度会压制有效步态的生成，导致前进缓慢。

- **role_id**: lateral_vertical_penalty  
  condition_to_use: 可在已有前进奖励时作为辅助，惩罚非前进方向的速度。  
  usable_signals: [body_y_velocity, body_z_velocity (obs[14], obs[15])]  
  risks: 极其微小的横向速度是四足步行固有的，过度惩罚会造成步态僵硬。

- **role_id**: angular_motion_penalty  
  condition_to_use: 若观察到翻滚或剧烈旋转时可单独启用，一般作为稳定化辅助项。  
  usable_signals: [body_roll_velocity, body_pitch_velocity, body_yaw_velocity (obs[16~18]), 各关节速度]  
  risks: 微小转动是不可避免的，需设置死区或平方窗。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: survival_bonus  
  reason: 环境没有存活奖励分量且无法获取终止标志，按时间步给予固定存活奖励会直接破坏前进任务的方向性，鼓励原地不动。  
  forbidden_or_missing_signals: [terminated, info['reward_survive']]

- **role_id**: goal_reaching  
  reason: 任务无固定目标点，距离或位置信息不可用，也无法定义成功，不应采用基于坐标的到达奖励。  
  forbidden_or_missing_signals: [x_position, y_position, distance_from_origin]

- **role_id**: contact_smoothness  
  reason: 接触力信息完全不可观测，无法设计基于足端接触的奖励。  
  forbidden_or_missing_signals: [contact_force]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | body_x_velocity | x_position, distance | dense_state_signal, linear_clip, soft_scaling | 可以直接返回值或经过缩放，需确保正值可解释。 |
| upright_posture | quat_x, quat_y (→ body_up_z) | torso_orientation_matrix | bounded_signal, linear_to_one_reward | 建议映射到接近1的奖励（如 0.5*(body_up_z+
