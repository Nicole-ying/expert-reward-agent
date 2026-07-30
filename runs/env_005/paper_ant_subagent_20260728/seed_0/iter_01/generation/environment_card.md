# 匿名环境理解卡片

## 1. 任务目标
控制一个3D四足机器人向前稳定行走/奔跑。核心目标是产生持续的前向速度，同时保持身体高度在安全范围（0.2 ~ 1.0）内不摔倒。次要目标包括维持直立姿态、减少侧向漂移、控制能耗和动作平滑。任务 **不要求** 到达某个指定位置，仅要求长期存活并向前移动。不能混淆为“仅站立不动”或“最小化能量消耗”，前进是刚性主目标。

## 2. 任务类型选择
selected_route_id: **locomotion_continuous_control**  
confidence: high  
reason: 核心目标是驱动四足机器人产生持续向前的运动，动力学是连续力矩控制的步行/跑步，属于典型的运动控制类任务。无到达点或明确终点，前进本身就是唯一主目标，其他高度、姿态是约束和次级优化项。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: 连续浮点数（具体精度由环境决定）
- obs[0] (body_z): 身体高度，reward_usable: true，可用作安全高度监控
- obs[1] (quat_w): 身体姿态四元数实部，reward_usable: true，参与直立度计算
- obs[2] (quat_x): 四元数虚部 x，reward_usable: true
- obs[3] (quat_y): 四元数虚部 y，reward_usable: true
- obs[4] (quat_z): 四元数虚部 z，reward_usable: true
- obs[5] (joint_1_angle): 髋关节1角度，reward_usable: true（可做动作平滑或参考姿态）
- obs[6] (joint_2_angle): 踝关节1角度，reward_usable: true
- obs[7] (joint_3_angle): 髋关节2角度，reward_usable: true
- obs[8] (joint_4_angle): 踝关节2角度，reward_usable: true
- obs[9] (joint_5_angle): 髋关节3角度，reward_usable: true
- obs[10] (joint_6_angle): 踝关节3角度，reward_usable: true
- obs[11] (joint_7_angle): 髋关节4角度，reward_usable: true
- obs[12] (joint_8_angle): 踝关节4角度，reward_usable: true
- obs[13] (body_x_velocity): 世界x轴（前向）速度，reward_usable: true，**主前向奖励信号**
- obs[14] (body_y_velocity): 世界y轴（侧向）速度，reward_usable: true，可惩罚侧向
- obs[15] (body_z_velocity): 垂直速度，reward_usable: true，可惩罚剧烈上下起伏
- obs[16] (body_roll_velocity): 滚转角速度，reward_usable: true，用于稳定性惩罚
- obs[17] (body_pitch_velocity): 俯仰角速度，reward_usable: true
- obs[18] (body_yaw_velocity): 偏航角速度，reward_usable: true，转弯惩罚
- obs[19] (joint_1_velocity): 关节1角速度，reward_usable: true（动作平滑/能耗）
- obs[20] (joint_2_velocity): 关节2角速度，reward_usable: true
- obs[21] (joint_3_velocity): 关节3角速度，reward_usable: true
- obs[22] (joint_4_velocity): 关节4角速度，reward_usable: true
- obs[23] (joint_5_velocity): 关节5角速度，reward_usable: true
- obs[24] (joint_6_velocity): 关节6角速度，reward_usable: true
- obs[25] (joint_7_velocity): 关节7角速度，reward_usable: true
- obs[26] (joint_8_velocity): 关节8角速度，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- 连续动作，每个维度范围 [[-1.0, 1.0]]
- action_dim 0: hip_1_torque — 第一髋关节扭矩
- action_dim 1: ankle_1_torque — 第一踝关节扭矩
- action_dim 2: hip_2_torque — 第二髋关节扭矩
- action_dim 3: ankle_2_torque — 第二踝关节扭矩
- action_dim 4: hip_3_torque — 第三髋关节扭矩
- action_dim 5: ankle_3_torque — 第三踝关节扭矩
- action_dim 6: hip_4_torque — 第四髋关节扭矩
- action_dim 7: ankle_4_torque — 第四踝关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无明确的成功终止标志；可默认为“在时间限制（truncation）内始终保持健康姿态”视为一次成功完整运行。
- **failure-like termination**:  
  - body_height_outside_healthy_range：身体高度 z ≤ 0.2（摔倒）或 z ≥ 1.0（过度跃起）。  
  - state_value_outside_finite_range：任何状态值变为 NaN 或 inf，通常代表物理崩溃。  
  两类均直接终止回合，属于硬失败。
- **ambiguous termination**: 无。
- **truncation**: time_limit_reached（达到最大步数），表示存活完全程。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** 
- explicit_failure_flag_available: **false** （`info` 字段为空，不能直接获得终止原因，仅能从环境返回的 `terminated` 或 `truncated` 在 RL 循环中判断，但奖励函数接口不提供这些标志）
- allowed_info_fields: 无（info 为空字典）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用**：
- `obs` (当前状态向量，27维)
- `action` (施加的动作向量，8维)
- `next_obs` (下一状态向量，27维)
- `info` 中明确允许的字段（当前允许字段为空）
- `training_progress` 只有任务提示明确允许时才使用（此处未明确允许，但可忽略）

**禁止使用**：
- `original_reward` （官方奖励被屏蔽）
- 任何被禁止的 info 字段：reward_*, x/y_position 等
- 未声明的 obs 切片（例如假想中有全局 x 坐标，实际上不存在）

## 7. 可用于奖励函数的信号
- **位置相关**：身体高度 `body_z`（obs[0]）；身体姿态四元数 `quat_w,x,y,z`（obs[1:5]），可计算 body_up_z。关节角度（obs[5:13]）可构造姿态正则化或对称性惩罚。
- **速度相关**：前向速度 `body_x_velocity`（obs[13]）——直接前进奖励；侧向速度 `body_y_velocity`（obs[14]）——侧向漂移惩罚；垂直速度 `body_z_velocity`（obs[15]）——起伏惩罚；角速度 `body_roll/pitch/yaw_vel`（obs[16:19]）——稳定性和转向惩罚；关节角速度（obs[19:27]）——动作平滑/能耗。
- **动作/执行器**：`action`（8维扭矩）可用于计算力矩大小、变化量。
- **其他**：训练进度（若环境描述明确需要，但此处未强调，谨慎使用）。

## 8. 不确定或不可用的信号
- 无全局 x/y 位置，无法奖励绝对前进距离。
- 无脚与地面接触力信息，无法使用接触软着陆或 gait pattern 奖励。
- 无显式 success/failure 标志，只能从观测预防。
- 无身体质量、惯性等物理参数，无法计算精确能耗（仅能估算扭矩×速度）。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped
  actuator_type: torque_controlled_joints (8 DOF)
  contact_structure: leg-ground contact (not observable)
primary_objectives:
  - forward_velocity: 最大化身体前向速度（或达到目标速度）
  - health: 保持身体高度在 (0.2, 1.0) 区间内，防止摔倒
  - upright: 保持躯干近乎垂直，投影 close to 1
secondary_objectives:
  - lateral_stability: 减小侧向速度
  - yaw_stability: 减小偏航角速度
  - energy_efficiency: 减小关节力矩和运动幅度
  - smoothness: 动作连续，避免高频震荡
main_failure_risks:
  - 高度过低（摔倒）或过高（失控跳跃）导致终止
  - 姿态过度倾斜（翻滚）导致物理崩溃
  - 前进速度停滞，长时间原地踏步
  - 动作剧烈震荡，能耗高且不稳定
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: forward_velocity**  
  purpose: 鼓励持续向前运动，主目标  
  why_required: 任务描述明确要求“must walk or run forward”，否则只是平衡站立  
  usable_signals: body_x_velocity (obs[13])  
  risks: 若权重过大可能导致机器人采取极端前倾姿态或不稳定步态；若仅正向奖励无上限可能导致非物理跳跃

- **role_id: body_height_safety**  
  purpose: 防止身体高度超出健康范围导致提前终止  
  why_required: 高度超出区间直接终止，必须作为硬惩罚或饱和惩罚来保障生存  
  usable_signals: body_z (obs[0])  
  risks: 仅用中心化吸引力可能无法提供足够的边界紧急惩罚，可能需要非对称分段惩罚（对过低区域更重）

- **role_id: upright_orientation**  
  purpose: 维持躯干竖直，防止侧翻或倒立  
  why_required: 连续大幅度倾斜会破坏前进并容易导致摔倒，且隐式帮助对称步态  
  usable_signals: quat (obs[1:5]) 计算的 body_up_z = 1 - 2*(quat_x² + quat_y²)  
  risks: 过度强调可能抑制正常步态中必要的轻微身体摆动，需留出一定容差

### 10.2 条件职责 conditional_roles
- **role_id: lateral_velocity_penalty**  
  condition_to_use: 当侧向速度 |body_y_velocity| 的经验数值明显偏离零（非偶然噪声）时加入。若早期就加入可能过度约束探索  
  usable_signals: body_y_velocity (obs[14])  
  risks: 可能限制初始随机探索时必要的侧向调整，建议较小权重或延迟激活

- **role_id: action_smoothness**  
  condition_to_use: 训练中后期，步态初步形成后，若观察到动作剧烈震荡时引入  
  usable_signals: action (当前动作) 与上一动作对比（需要状态存储，但 reward 接口可访问上一动作？只能在 RL 循环中实现，或利用 next_obs 与 action 结合？