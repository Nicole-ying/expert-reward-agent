# 匿名环境理解卡片

## 1. 任务目标
本环境要求一个 3D 四足机器人通过 8 个连续扭矩关节控制前进，核心目标是**持续向前行走或奔跑**。在前进过程中，机器人必须保持身体高度在健康范围内（0.2 ~ 1.0 m，用 body_z 表示），并尽量保持直立姿态。若高度超出范围或任意状态量变为无效值，回合立即终止。机器人需要**同时兼顾前进速度、身体高度维持和姿态稳定性**，而不是仅保持站立不动。由于无法获取全局 x/y 位置，只能基于瞬时前向速度进行闭环控制，不得依赖累积距离。

## 2. 任务类型选择
selected_route_id: `locomotion_continuous_control`  
confidence: high  
reason: 核心目标是持续前进通过地形（walk or run forward），高度约束是健康的必要条件而非唯一目标，且不存在导航目标点、抓取操作或稀疏探索事件，完全属于典型的多足连续运动控制。

## 3. 观察空间 observation_space
- type: Box  
- shape: (27,)  
- dtype: float32  
- 每个维度含义：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|--------------|
| 0 | body_z | 身体中心在世界 z 轴的高度 | true |
| 1 | quat_w | 身体姿态四元数实部 w | true |
| 2 | quat_x | 四元数虚部 x | true |
| 3 | quat_y | 四元数虚部 y | true |
| 4 | quat_z | 四元数虚部 z | true |
| 5 | joint_1_angle | 第 1 髋关节角度 | true |
| 6 | joint_2_angle | 第 1 踝关节角度 | true |
| 7 | joint_3_angle | 第 2 髋关节角度 | true |
| 8 | joint_4_angle | 第 2 踝关节角度 | true |
| 9 | joint_5_angle | 第 3 髋关节角度 | true |
| 10 | joint_6_angle | 第 3 踝关节角度 | true |
| 11 | joint_7_angle | 第 4 髋关节角度 | true |
| 12 | joint_8_angle | 第 4 踝关节角度 | true |
| 13 | body_x_velocity | 世界系 x 方向线速度（前向） | true |
| 14 | body_y_velocity | 世界系 y 方向线速度（横向） | true |
| 15 | body_z_velocity | 世界系 z 方向线速度（竖直） | true |
| 16 | body_roll_velocity | 绕 x 轴角速度 | true |
| 17 | body_pitch_velocity | 绕 y 轴角速度 | true |
| 18 | body_yaw_velocity | 绕 z 轴角速度 | true |
| 19 | joint_1_velocity | 第 1 髋关节角速度 | true |
| 20 | joint_2_velocity | 第 1 踝关节角速度 | true |
| 21 | joint_3_velocity | 第 2 髋关节角速度 | true |
| 22 | joint_4_velocity | 第 2 踝关节角速度 | true |
| 23 | joint_5_velocity | 第 3 髋关节角速度 | true |
| 24 | joint_6_velocity | 第 3 踝关节角速度 | true |
| 25 | joint_7_velocity | 第 4 髋关节角速度 | true |
| 26 | joint_8_velocity | 第 4 踝关节角速度 | true |

## 动作空间 action_space
- type: Box  
- shape: [8]  
- continuous 动作，每个维度范围 [-1.0, 1.0]

| 动作维度 | 名称 | 含义 |
|----------|------|------|
| 0 | hip_1_torque | 第 1 髋关节扭矩 |
| 1 | ankle_1_torque | 第 1 踝关节扭矩 |
| 2 | hip_2_torque | 第 2 髋关节扭矩 |
| 3 | ankle_2_torque | 第 2 踝关节扭矩 |
| 4 | hip_3_torque | 第 3 髋关节扭矩 |
| 5 | ankle_3_torque | 第 3 踝关节扭矩 |
| 6 | hip_4_torque | 第 4 髋关节扭矩 |
| 7 | ankle_4_torque | 第 4 踝关节扭矩 |

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: 无显式成功终止；environment 仅通过时间截断 `truncated` 结束，此时若机器人在高度边界内且不崩溃，可视为存活成功，但不构成硬 success flag。
- failure-like termination:  
  - `body_height_outside_healthy_range`: 身体高度 body_z ≤ 0.2 或 ≥ 1.0，表示摔倒/弹飞。  
  - `state_value_outside_finite_range`: 任一状态量变为 NaN 或 Inf，这是数值崩溃。  
- ambiguous termination: 无。  
- truncation: 达到时间上限，是唯一正常结束方式。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: []  (info 为空字典，无任何字段)  
- forbidden_or_uncertain_info_fields:  
  - `reward_forward`, `reward_ctrl`, `reward_contact`, `reward_survive` 等官方 reward 项（禁止）  
  - `x_position`, `y_position`, `distance_from_origin` 等位置信息（禁止）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs` (当前步 27 维观测)  
- `action` (当前步输出 8 维扭矩)  
- `next_obs` (下一步 27 维观测)  
- `info` 中经本卡片允许的字段（当前为无）  
- `training_progress` 仅在 prompt 明确允许时用作调度因子，不得作为主要信号

禁止使用：
- `original_reward`（掩码且明确禁止）  
- 任何未声明的 `info` 字段  
- 任何未在本卡片声明的 obs 切片（如假设存在隐藏的 x/y 坐标）

## 7. 可用于奖励函数的信号
- position:  
  - `body_z` (obs[0])，身体高度  
  - 四元数姿态 (obs[1:5])，可计算 `body_up_z = 1 - 2*(quat_x² + quat_y²)` 得到躯干对世界 z 轴的投影（1.0 为完全直立）  
  - 8 个关节角度 (obs[5:13])
- velocity:  
  - 前向速度 `body_x_velocity` (obs[13])  
  - 横向速度 `body_y_velocity` (obs[14])  
  - 竖直速度 `body_z_velocity` (obs[15])  
  - 角速度 `body_roll/pitch/yaw_velocity` (obs[16-18])  
  - 8 个关节角速度 (obs[19:27])
- orientation: 可由 obs[1:5] 推导 body_up_z 等
- contact: **无接触力信息**，无法使用
- action/engine:  
  - 动作向量 `action` (8D) 本身，可用于扭矩惩罚或平滑惩罚  
  - 若需动作变化，需在外部保存上一动作状态，但不鼓励在无状态奖励函数中引入全局记忆

## 8. 不确定或不可用的信号
- 全局 x 坐标、y 坐标及累积前进距离（未提供，不可用）
- 任何官方 reward 分量（被掩码且禁止）
- 接触力、触地标志、脚部滑移等（未提供）
- 成功/失败标志位（info 为空）
- 直立角度 θ 的 sin/cos 形式（需要自行转换四元数）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped (四足，每条腿有髋-踝两个关节)
  actuator_type: torque-controlled (8 个连续力矩关节)
  contact_structure: 足式，有地面接触但无接触力信息
primary_objectives:
  - 最大化前进速度 (body_x_velocity)
  - 保持身体高度在 (0.2, 1.0) 范围内，避免摔倒或弹跳过高
  - 维持躯干直立（body_up_z 尽量接近 1.0）
secondary_objectives:
  - 抑制横向运动和偏航，保持直线前进
  - 降低关节力矩（能耗）
  - 平稳动作，避免高频抖动
main_failure_risks:
  - 向前倾斜摔倒（高度过低）
  - 过度依赖后腿导致前腿抬起过高甚至翻滚
  - 侧向漂移大，偏离前进方向
  - 高扭矩快速切换导致关节震荡或数值溢出
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: `forward_velocity`
  - purpose: 鼓励持续向前运动，这是任务的核心目标
  - why_required: 不加入前进激励，策略只会学习站立或极慢运动
  - usable_signals: [`body_x_velocity` (obs[13])]
  - risks: 若权重过高可能导致忽视高度约束而摔倒；需与 height 角色配合

- role_id: `height_survival`
  - purpose: 惩罚身体高度接近或超出边界，维持安全范围
  - why_required: 高度是终止条件，必须让策略学会在高度内运动
  - usable_signals: [`body_z` (obs[0])]
  - risks: 仅使用一个硬边界值可能导致稀疏梯度；推荐用中心区域奖励 + 边界惩罚

- role_id: `upright_orientation`
  - purpose: 使躯干尽量朝上，防止翻滚
  - why_required: 如果不约束姿态，四足可能倒立或侧翻完成前进（不稳定）
  - usable_signals: [四元数 obs[1:5]，可计算 `body_up_z`]
  - risks: 四足前进时允许轻微前倾，过度惩罚可能抑制步态；应允许 body_up_z 在 0.8~1.0 间自由

### 10.2 条件职责 conditional_roles
- role_id: `lateral_and_yaw_penalty`
  - purpose: 抑制横向速度和偏航角速度，保证直线前进
  - condition_to_use: 观察到策略出现明显侧向漂移或原地打转时启用；也可作为默认软约束
  - usable_signals: [`body_y_velocity` (obs[14]), `body_yaw_velocity` (obs[18])]
  - risks: 过强约束可能限制自然步态的轻微横向摆动，应使用小系数

- role_id: `action_smoothness`
  - purpose: 减少相邻动作的剧烈变化，降低关节抖动
  - condition_to_use: 若动作出现高频振荡，可从训练中期加入（需存储上一动作，或通过joint velocity间接约束）
  - usable_signals: [action (当前帧，需与上一帧比较)，或使用关节角速度 (obs[19:27]) 作为代理]
  - risks: 需要存储 history，若使用关节角速度平方则可避免状态，但惩罚的是速度而非加速度变化

- role_id: `torque_cost`
  - purpose: 降低能耗，鼓励高效步态
  - condition_to_use: 通常在训练后期或作为弱正则项加入，以免干扰主要目标
  - usable_signals: [action (8D 扭矩)]
  - risks: 过大的惩罚会导致机器人无力前进，必须极低权重

### 10.3 慎用/禁用职责 avoid_roles
- role_id: `global_progress_reward`
  - reason: 需要 x 或 y 位置增量，但观测中缺失，无法实现
  - forbidden_or_missing_signals: [x_position, y_position]

- role_id: `contact_force_penalty`
  - reason: 接触力信息未提供，无法计算足端受力
  - forbidden_or_m