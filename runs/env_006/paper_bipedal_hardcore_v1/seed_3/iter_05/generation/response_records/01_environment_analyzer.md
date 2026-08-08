# Response Record

# 匿名环境理解卡片

## 1. 任务目标
双足机器人在布满梯子、树桩、坑洼等的不规则崎岖地形上向前行进，目标是走得尽可能远并在不摔倒的前提下到达地形尽头。同时，需要尽量减少不必要的关节力矩以避免能源浪费。机器人通过 LIDAR 传感器提前感知前方地形起伏，从而在接近障碍物前调整步态。**主目标**是持续稳定地向前移动（最大化前进距离），**次目标**是保持平衡避免摔倒以及降低关节能耗。不应将“到达尽头”视为唯一的点型成功目标，而应把它看作过程性前进的自然终结。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心目标不受特定目标位置约束，而是尽可能在困难地形上持续前进；附属目标包括防摔倒和降低能耗，都属于连续移动任务的常见额外偏好，并不构成等价的多目标冲突，因此不属于 multi_objective_task。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（隐含，从描述推断为连续数值）
- obs[0]: hull_angle, 机体俯仰角(rad)，reward_usable: true
- obs[1]: hull_angular_velocity, 机体俯仰角速度(rad/s)，reward_usable: true
- obs[2]: horizontal_speed, 机体水平前进速度(m/s)，reward_usable: true
- obs[3]: vertical_speed, 机体垂直速度(m/s)，reward_usable: true
- obs[4]: joint_0_angle, 髋关节1角度(rad)，reward_usable: true（可做姿态约束）
- obs[5]: joint_0_speed, 髋关节1角速度(rad/s)，reward_usable: true
- obs[6]: joint_1_angle, 膝关节1角度(rad)，reward_usable: true
- obs[7]: joint_1_speed, 膝关节1角速度(rad/s)，reward_usable: true
- obs[8]: joint_2_angle, 髋关节2角度(rad)，reward_usable: true
- obs[9]: joint_2_speed, 髋关节2角速度(rad/s)，reward_usable: true
- obs[10]: joint_3_angle, 膝关节2角度(rad)，reward_usable: true
- obs[11]: joint_3_speed, 膝关节2角速度(rad/s)，reward_usable: true
- obs[12]: leg_1_ground_contact, 腿1触地标志（0/1），reward_usable: true（可用于检测支撑相）
- obs[13]: leg_2_ground_contact, 腿2触地标志（0/1），reward_usable: true
- obs[14]: lidar_1, 前方地形高度测量1 (m)，reward_usable: false（高程信息难以直接翻译成标量奖励，除非有复杂地形适应目标）
- obs[15]: lidar_2, reward_usable: false
- obs[16]: lidar_3, reward_usable: false
- obs[17]: lidar_4, reward_usable: false
- obs[18]: lidar_5, reward_usable: false
- obs[19]: lidar_6, reward_usable: false
- obs[20]: lidar_7, reward_usable: false
- obs[21]: lidar_8, reward_usable: false
- obs[22]: lidar_9, reward_usable: false
- obs[23]: lidar_10, reward_usable: false

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- bounds: [-1.0, 1.0]
- action[0]: hip_1_torque, 髋关节1力矩（归一化）
- action[1]: knee_1_torque, 膝关节1力矩
- action[2]: hip_2_torque, 髋关节2力矩
- action[3]: knee_2_torque, 膝关节2力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `reached_end_of_terrain`（到达地形尽头），可视为成功完成移动任务。
- failure-like termination: `body_fallen_over`（机体摔倒），应被视为失败。
- ambiguous termination: 无。只有上述两种明确终止。
- truncation: 无额外时间截断说明，但往往有 episode 长度上限，未明确说明，在任务分析中暂不作为独立终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无法直接读取到达尽头标志）
- explicit_failure_flag_available: false（info 为空，无法直接读取摔倒标志）
- allowed_info_fields: []（空列表，所有 info 字段不可用）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

终止类型必须通过观测信号间接推断：
- **摔倒可能推断路径**：`hull_angle` 的绝对值急剧增大（例如超过 0.5~1.0 rad），或 `hull_angular_velocity` 在终止前骤增；同时 `leg_1_ground_contact` 和 `leg_2_ground_contact` 同时消失（腿离地）可作为辅助证据。记为 **derived_possible**。
- **到达尽头可能推断路径**：episode 提前结束且观测并未出现上述摔倒特征，尤其是 `horizontal_speed` 从前一时刻的正常前进突然终止，可视为达到尽头。也记为 **derived_possible**。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action
- next_obs
- info 中明确允许的字段（此处 info 为空，故**不可用**任何 info 字段）
- training_progress 仅在后续明确指示允许时可用（当前任务未指明，应视为**禁用**）

禁止使用：
- original_reward
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片（例如不能假设存在不在观测列表中的绝对位置）

## 7. 可用于奖励函数的信号
以下信号可从 obs、next_obs、action 及 derived_possible 推断中获得：
- position: 无绝对位置可直接使用（没有 x 坐标）
- velocity: 
  - next_obs[2] (horizontal_speed) —— **直接可用**
  - next_obs[3] (vertical_speed) —— 可用作平滑惩罚
- orientation:
  - next_obs[0] (hull_angle) —— **直接可用**，惩罚过大倾斜
  - next_obs[1] (hull_angular_velocity) —— 直接可用，惩罚急剧转动
- contact:
  - next_obs[12] (leg_1_ground_contact)
  - next_obs[13] (leg_2_ground_contact)
  可用于检测腿是否着地，或奖励交替支撑。
- action/engine:
  - action[0..3]（关节力矩）—— **直接可用**，惩罚能耗
- other:
  - **derived_possible: fall_detected** —— 从 hull_angle 和 angular_velocity 超过阈值推断摔倒
  - **derived_possible: end_reached** —— 推断到达尽头，但风险较高，一般不单独作为奖励信号，而由外部截断终止处理。

## 8. 不确定或不可用的信号
- 绝对位置 x, y, z 不存在于观测中
- 地形高度真值（只有相对 lidar 测量，无法直接得 reward）
- 成功/失败标志不存在（info 空）
- training_progress 当前不可用
- 能量消耗的积分量不存在
- 目标速度或目标位置不存在

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_rough_terrain_gait
control_type: continuous
morphology:
  body_type: bipedal_planar_body
  actuator_type: torque_controlled_joints (2 hips, 2 knees)
  contact_structure: two_feet_ground_contact
primary_objectives:
  - maximize_forward_progress_on_rough_terrain
secondary_objectives:
  - avoid_falling
  - minimize_joint_torque_energy
main_failure_risks:
  - falling_over_due_to_obstacle_impact_or_imbalance
  - getting_stuck_and_not_moving_forward
  - excessive_energy_consumption_without_progress
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 激励 agent 向前移动
  why_required: 任务核心是走得远，主 reward 必须驱使前进
  usable_signals: [next_obs[2] horizontal_speed]
  risks: 可能导致 agent 忽略平衡，摔倒时仍追求速度；需要配合 fall_prevention

- role_id: fall_prevention
  purpose: 抑制导致摔倒的姿态
  why_required: 一旦摔倒 episode 即终止，无法继续前进；必须通过惩罚危险姿态来保护 agent
  usable_signals: [next_obs[0] hull_angle, next_obs[1] hull_angular_velocity, derived_possible fall_detected]
  risks: 过强的惩罚可能使 agent 不敢迈步，导致保守的原地踏步

- role_id: energy_efficiency
  purpose: 降低不必要的大关节力矩，节约能量
  why_required: 任务明确要求 minimize unnecessary joint torque
  usable_signals: [action[0..3]]
  risks: 若权重过大，可能迫使 agent 用极小动作，导致前进不足

### 10.2 条件职责 conditional_roles
- role_id: contact_symmetry
  condition_to_use: 当 agent 已经有初步前进能力后，可鼓励双腿交替支撑以形成自然步态，避免单腿过度依赖
  usable_signals: [next_obs[12], next_obs[13] leg ground contact indicators]
  risks: 初期加入可能干扰探索，因为稳定步态尚未形成

- role_id: vertical_smoothness
  condition_to_use: 当地形过于崎岖时，加入轻微惩罚垂直速度剧烈波动，可提高运动平稳性
  usable_signals: [next_obs[3] vertical_speed]
  risks: 过度平滑可能阻碍 agent 通过障碍所需的跳跃动作

### 10.3 慎用/禁用职责 avoid_roles
- role_id: alive_bonus
  reason: 简单生存奖励会鼓励原地待命而不前进，与主目标冲突
  forbidden_or_missing_signals: 无对应终止信号可度量生存

- role_id: target_velocity_tracking
  reason: 没有预设的目标前进速度；此环境要求尽可能远而不是恒定速度，不应使用
  forbidden_or_missing_signals: 缺少目标速度值

- role_id: lidar_based_terrain_penalty
  reason: 奖励函数中很难将 lidar 高程图直接翻译成标量惩罚项，若设计不当反而会引入噪声；可在后续迭代中考虑，但当前不建议
  forbidden_or_missing_signals: 无法将多维度 lidar 信号简化为合理的标量奖励

## 11. role_to_signal_mapping
| role_id | usable
