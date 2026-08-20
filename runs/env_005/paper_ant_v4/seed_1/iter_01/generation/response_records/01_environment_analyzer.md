# Response Record

# 匿名环境理解卡片

## 1. 任务目标
学习一个连续控制的四足机器人（8个扭矩控制关节）在3D空间中稳定向前行走或跑动。任务的核心是产生持续的前向平移，同时保持身体高度在健康区间（0.2~1.0）且姿态基本直立。任何因为倒地、弹飞或数值异常导致的提前终止都视为失败。策略不能仅仅保持静止不倒，必须展现出有效的前进运动。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 主目标是持续前进通过地形（等价于无限长平面上的持续位移），次目标为维持躯干高度和姿态，符合 locomotion_continuous_control 的定义。该环境没有导航目标、不需要抓取、无需稀疏探索，也不是严格的多目标冲突任务，前进是唯一核心进展指标。

动力学子类型 dynamics_subtype: multi_legged_body_locomotion  
（四足多关节身体沿目标方向前进，带有地面接触和姿态约束）

## 3. 观察空间 observation_space
- type: Box  
- shape: [27]  
- dtype: float32（推断）  
- 各维度含义与奖励可用性：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | body_z | 躯干质心的世界 z 坐标（高度） | true |
| 1 | quat_w | 方向四元数实部 w | true（用于计算姿态） |
| 2 | quat_x | 方向四元数虚部 x | true |
| 3 | quat_y | 方向四元数虚部 y | true |
| 4 | quat_z | 方向四元数虚部 z | true |
| 5 | joint_1_angle | 第1髋关节角 | true（位置信号） |
| 6 | joint_2_angle | 第1踝关节角 | true |
| 7 | joint_3_angle | 第2髋关节角 | true |
| 8 | joint_4_angle | 第2踝关节角 | true |
| 9 | joint_5_angle | 第3髋关节角 | true |
| 10 | joint_6_angle | 第3踝关节角 | true |
| 11 | joint_7_angle | 第4髋关节角 | true |
| 12 | joint_8_angle | 第4踝关节角 | true |
| 13 | body_x_velocity | 躯干在世界 x 方向（前方）的速度 | true **← 核心前进信号** |
| 14 | body_y_velocity | 躯干在世界 y 方向（侧向）的速度 | true |
| 15 | body_z_velocity | 躯干在 z 方向的速度 | true |
| 16 | body_roll_velocity | 绕 x 轴角速度（翻滚） | true |
| 17 | body_pitch_velocity | 绕 y 轴角速度（俯仰） | true |
| 18 | body_yaw_velocity | 绕 z 轴角速度（偏航） | true |
| 19 | joint_1_velocity | 第1髋关节角速度 | true |
| 20 | joint_2_velocity | 第1踝关节角速度 | true |
| 21 | joint_3_velocity | 第2髋关节角速度 | true |
| 22 | joint_4_velocity | 第2踝关节角速度 | true |
| 23 | joint_5_velocity | 第3髋关节角速度 | true |
| 24 | joint_6_velocity | 第3踝关节角速度 | true |
| 25 | joint_7_velocity | 第4髋关节角速度 | true |
| 26 | joint_8_velocity | 第4踝关节角速度 | true |

额外可计算信号：body_up_z = 1 - 2*(quat_x² + quat_y²)，表示躯干 z 轴在世界 z 轴的投影（1.0 为完全直立）。

## 4. 动作空间 action_space
- type: Box  
- shape: [8]  
- continuous: true  
- 界限: [-1.0, 1.0]（每个关节扭矩）  

| 动作维度 | 名称 | 含义 |
|----------|------|------|
| 0 | hip_1_torque | 第1髋关节扭矩 |
| 1 | ankle_1_torque | 第1踝关节扭矩 |
| 2 | hip_2_torque | 第2髋关节扭矩 |
| 3 | ankle_2_torque | 第2踝关节扭矩 |
| 4 | hip_3_torque | 第3髋关节扭矩 |
| 5 | ankle_3_torque | 第3踝关节扭矩 |
| 6 | hip_4_torque | 第4髋关节扭矩 |
| 7 | ankle_4_torque | 第4踝关节扭矩 |

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **无明确成功终止**。正常存活到时间上限（truncation）可视为一种成功存活信号，但未保证前进距离。
- failure-like termination: terminated=True 由以下条件触发：
  - body_height_outside_healthy_range: body_z ≤ 0.2 或 body_z ≥ 1.0（倒塌/弹飞）
  - state_value_outside_finite_range: 任何状态值变为 NaN 或无穷大
- ambiguous termination: 无。
- truncation: 时间上限到达，truncated=True。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 被屏蔽，无法获取任何标志）
- explicit_failure_flag_available: false （同上，但可从 terminated 原因推断失败）
- allowed_info_fields: [] （空列表，意味着 **info 完全不可用**）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止，包括 reward_forward、reward_ctrl、reward_contact、reward_survive、x_position、y_position、distance_from_origin 等。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用的输入：
- `obs`：当前状态向量（27维）
- `action`：当前执行的动作向量（8维）
- `next_obs`：下一状态向量（27维）
- `training_progress`：**仅在 prompt 明确允许时使用，当前未明确允许，默认不使用**

严格禁止使用：
- `original_reward`（已屏蔽的官方奖励）
- `info` （任何字段均不可用）
- 任何未声明的全局变量或环境内部状态
- 从 `obs` 中投影出绝对 x/y 位置（环境未提供 x, y 坐标）

## 7. 可用于奖励函数的信号
- position: body_z（高度），关节角度（8个），身体姿态四元数（可计算 body_up_z）
- velocity: body_x_velocity（前向速度），body_y_velocity（侧向漂移），body_z_velocity（上下），body_roll/pitch/yaw_velocity（姿态变化率），8个关节速度
- orientation: 通过四元数计算身体直立度（body_up_z），或惩罚较大滚动/俯仰角速度
- action/engine: 8维扭矩动作（可用于动作幅度惩罚或平滑性约束）
- other: 从姿态推导的 body_up_z，身体高度距离健康边界的距离

## 8. 不确定或不可用的信号
- 绝对 x, y 位置：完全不可用（info 被屏蔽，且状态中未包含）
- 前进距离：无法累积，仅能通过瞬时前向速度间接鼓励
- 接触力：无接触传感器信息
- 存活奖励原语：info 中不存在，不能直接给予存活 bonus
- 成功率/失败标志：无显式 flag，只能依据 terminated/truncated 在外部评估时推断

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped
  actuator_type: torque-controlled joints (hip + ankle ×4)
  contact_structure: feet in contact with ground (not instrumented)
primary_objectives:
  - sustain positive forward velocity (body_x_velocity > 0)
  - maintain body height within (0.2, 1.0) at all times
  - keep torso upright (body_up_z close to 1.0)
secondary_objectives:
  - minimize lateral drift (body_y_velocity ≈ 0)
  - smooth energy-efficient actuation (small/coherent torques)
main_failure_risks:
  - falling over or collapsing (height → 0.2)
  - unexpectedly jumping/launching (height → 1.0)
  - excessive tumbling (high roll/pitch velocity, body_up_z small)
  - numerical instability (NaN/Inf in state)
  - progressing without control (e.g., falling forward passively but not robustly)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress  
  purpose: 鼓励机器人持续产生正向前进速度  
  why_required: 任务核心是行走/跑动，静止不倒不满足要求  
  usable_signals: [body_x_velocity (obs[13])]  
  risks: 若奖励过于线性，可能鼓励疯狂向前翻滚；需要与稳定职责组合

- role_id: health_height  
  purpose: 迫使身体高度维持在 (0.2, 1.0) 的安全区间内，避免终止  
  why_required: 高度越界直接导致 episode 失败  
  usable_signals: [body_z (obs[0])]  
  risks: 过于严厉的边界惩罚可能抑制探索；可用平滑的 soft boundary cost

- role_id: upright_stability  
  purpose: 保持躯体直立，防止严重倾斜或翻滚  
  why_required: 翻滚会导致高度丢失和不可控行为  
  usable_signals: [body_up_z (computed from quat), body_roll_velocity, body_pitch_velocity]  
  risks: 若只惩罚角度而不考虑角速度，可能引入抖动；需与前进目标共存

### 10.2 条件职责 conditional_roles
- role_id: lateral_drift_penalty  
  condition_to_use: 当观察到明显侧向漂移（|body_y_velocity|较大）且前进速度已经产生时，加入小惩罚以提升直线性  
  usable_signals: [body_y_velocity (obs[14])]  
  risks: 早期训练加入可能限制探索，应延迟或使用极小的系数

- role_id: action_magnitude_regularization  
  condition_to_use: 训练后期（或前进已较稳定后）引入，以鼓励节能和动作平滑  
  usable_signals: [action (8维)]  
  risks: 早期施加会抑制学习有效的步态，应逐步增加权重

### 10.3 慎用/禁用职责 avoid_roles
- role_id: survival_bonus  
  reason: 直接生存奖励通常需要 info 中的存活信号，当前环境没有；只能用高度健康隐式表达生存  
  forbidden_or_missing_signals: [survive reward, time_alive flag]

- role_id: goal_distance_progress  
  reason: 没有任何绝对位置信息，无法计算与目标的距离  
  forbidden_or_missing_signals: [x_position, y_position, distance_from_origin]

- role_id: contact_force_penalty  
  reason: 环境中没有接触力传感器  
  forbidden_or_missing_signals: [contact_forces]

- role_id: posture_mimicry  
  reason: 无参考轨迹或模仿目标，且早期自行发现自然步态更鲁棒  
  forbidden_or_missing_signals: [reference joint angles]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | body_x_velocity (obs[13]) | absolute x displacement | bounded_signal (e.g., clip or tanh), linear reward for positive velocity | 可乘以 height
