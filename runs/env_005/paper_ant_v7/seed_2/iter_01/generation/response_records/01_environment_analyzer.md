# Response Record

# 匿名环境理解卡片

## 1. 任务目标
这是一个连续控制运动任务。控制一个 3D 四足机器人（8 个扭矩关节）在保证身体高度处于健康区间、保持直立姿态的前提下，以尽可能快的速度稳定前进。核心目标是稳定的前进运动（**前进速度最大化**），维持身体高度和直立姿态是避免提前终止的必要条件，但不是任务本身的核心优化目标。不允许依赖任何官方奖励项（info 被清空，official reward masked）。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务描述明确指出“必须行走或奔跑前进”，核心目标是“稳定的前进运动”，而不是纯存活或平衡；附属的高度/姿态约束仅作为终止条件存在，并非平行多目标。典型的连续动作运动控制任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: 推断为 float32（来自 continuous locomotion）
- obs[0] (body_z)：身体重心垂直高度。reward_usable: true（可构建高度健康奖励/惩罚）
- obs[1] (quat_w)：身体姿态四元数实部 w。reward_usable: true（用于计算直立程度）
- obs[2] (quat_x)：姿态四元数虚部 x。reward_usable: true
- obs[3] (quat_y)：姿态四元数虚部 y。reward_usable: true
- obs[4] (quat_z)：姿态四元数虚部 z。reward_usable: true
- obs[5] (joint_1_angle)：第一 hip 关节角度。reward_usable: true（可用来约束关节范围、平滑动作）
- obs[6] (joint_2_angle)：第一 ankle 关节角度。reward_usable: true
- obs[7] (joint_3_angle)：第二 hip 关节角度。reward_usable: true
- obs[8] (joint_4_angle)：第二 ankle 关节角度。reward_usable: true
- obs[9] (joint_5_angle)：第三 hip 关节角度。reward_usable: true
- obs[10] (joint_6_angle)：第三 ankle 关节角度。reward_usable: true
- obs[11] (joint_7_angle)：第四 hip 关节角度。reward_usable: true
- obs[12] (joint_8_angle)：第四 ankle 关节角度。reward_usable: true
- obs[13] (body_x_velocity)：身体在世界系 x 方向的前进速度。reward_usable: **true（核心前进信号）**
- obs[14] (body_y_velocity)：身体横向速度（世界 y）。reward_usable: true（可用于惩罚侧向漂移）
- obs[15] (body_z_velocity)：身体垂直速度。reward_usable: true（可用于惩罚剧烈上下颠簸）
- obs[16] (body_roll_velocity)：滚转角速度。reward_usable: true
- obs[17] (body_pitch_velocity)：俯仰角速度。reward_usable: true
- obs[18] (body_yaw_velocity)：偏航角速度。reward_usable: true
- obs[19] (joint_1_velocity)：第一 hip 关节角速度。reward_usable: true（用于平滑或能耗惩罚）
- obs[20] (joint_2_velocity)：第一 ankle 关节角速度。reward_usable: true
- obs[21] (joint_3_velocity)：第二 hip 关节角速度。reward_usable: true
- obs[22] (joint_4_velocity)：第二 ankle 关节角速度。reward_usable: true
- obs[23] (joint_5_velocity)：第三 hip 关节角速度。reward_usable: true
- obs[24] (joint_6_velocity)：第三 ankle 关节角速度。reward_usable: true
- obs[25] (joint_7_velocity)：第四 hip 关节角速度。reward_usable: true
- obs[26] (joint_8_velocity)：第四 ankle 关节角速度。reward_usable: true

## 4. 动作空间 action_space
- type: Box（连续）
- shape: (8,)
- 范围：[-1.0, 1.0] per joint（扭矩归一化值）
- action_dim 0 (hip_1_torque)：第一 hip 关节扭矩
- action_dim 1 (ankle_1_torque)：第一 ankle 关节扭矩
- action_dim 2 (hip_2_torque)：第二 hip 关节扭矩
- action_dim 3 (ankle_2_torque)：第二 ankle 关节扭矩
- action_dim 4 (hip_3_torque)：第三 hip 关节扭矩
- action_dim 5 (ankle_3_torque)：第三 ankle 关节扭矩
- action_dim 6 (hip_4_torque)：第四 hip 关节扭矩
- action_dim 7 (ankle_4_torque)：第四 ankle 关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **无明显成功终止**。episode 如果一直保持安全状态直到被截断（truncation）则可能被视为成功完成一次稳定的前进回合。
- failure-like termination: 身体高度低于 0.2（跌倒）或高于 1.0（过度起跳）；任何状态值变为 NaN 或 inf（数值崩溃）。
- ambiguous termination: 无。
- truncation: 达到环境预设的最大仿真步数（时间限制），此时 episode 直接结束，无特殊终止标志。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false（terminated 信号在 step 外部返回，但 reward 函数接口无法直接获取 terminated 标志）
- allowed_info_fields: **无**（info 字典被清空，接口声明禁止使用任何 info 字段）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等全部官方奖励或定位数据（明确禁止）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs：当前状态（27维向量）
- action：当前动作（8维向量）
- next_obs：下一状态（27维向量）
- **training_progress**：仅在明确声明允许时使用（本任务 prompt 未说明，默认不应依赖）
- **info**：始终为空字典 `{}`，不得使用任何字段

禁止使用：
- original_reward（被掩码的官方奖励，接口要求禁止）
- 任何从 info 中非法获取的字段（包括但不限于 forward, ctrl, survive 等）
- 任何未声明的 obs 切片（以本卡片定义为准）

## 7. 可用于奖励函数的信号
- position: body_z（高度），关节角度（可通过与目标姿态的偏差设计奖励）
- velocity: body_x_velocity（前进速度，核心），body_y_velocity（侧向），body_z_velocity（垂直速度），各关节角速度
- orientation: body_up_z = 1 - 2*(quat_x² + quat_y²) 量化直立程度（0~1，1 为完全竖直）
- contact: 无直接接触力，本环境版本无接触信息
- action/engine: action 本身（扭矩可构成能量/平滑惩罚），action 变化量（需自行维护上次动作，但奖励函数无状态，故无法直接计算 delta；可以惩罚 action 的绝对大小）
- other: 关节角度偏离正常范围（如设定目标关节位置）可用作风格约束

## 8. 不确定或不可用的信号
- 绝对世界坐标（x_position, y_position）被禁止，无法用于计算全程位移
- 接触力、足端触地标志、地面反作用力均不可用
- 成功/失败标志不可从 info 获取，也无法从 terminated 直接传入 reward 函数
- 任何官方奖励分量（forward, ctrl, contact, survive）均不可用
- 步态周期事件（如足着地瞬间）未提供信号，不建议依赖
- 上一帧动作不可在 stateless reward 中直接使用，无法计算动作变化率

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: 3D quadruped（4 条腿，每条腿 hip + ankle 两关节）
  actuator_type: torque_controlled (8 个独立扭矩，范围[-1,1])
  contact_structure: foot-ground contact masked（无接触力信息）
primary_objectives:
  - 最大化向前速度（body_x_velocity），维持高效前进步伐
secondary_objectives:
  - 保持身体高度在安全区间 (0.2~1.0)，避免早停
  - 维持直立姿态（body_up_z 接近 1）
  - 减小侧向漂移与垂直跳动
  - 动作平滑且节能（小扭矩、小关节速度）
main_failure_risks:
  - 因重心过低（<0.2）摔倒终止
  - 过度跳跃导致高度超过 1.0 终止
  - 关节发力过大引发数值不稳定（NaN/inf）
  - 策略陷入静止不动或原地踏步（forward velocity ~0），虽不终止但无意义
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity_reward  
  purpose: 直接鼓励身体在世界 x 方向的前进速度  
  why_required: 是任务唯一明确的核心目标（“行走或奔跑前进”），所有其他职责均为辅助  
  usable_signals: [next_obs[13] (body_x_velocity)]  
  risks: 若权重过高可能导致策略忽略稳定性，引发早停；静止/后退策略会获得负或零奖励，需要保证梯度指向期望方向

### 10.2 条件职责 conditional_roles
- role_id: healthy_height_survival  
  purpose: 鼓励身体高度保持在安全区间中部，远离早停边界  
  condition_to_use: 当高度接近 (0.2, 1.0) 边界时给予惩罚，中部给予小奖励或零惩罚；可全程开启但需低频权重，避免与前进速度冲突  
  usable_signals: [next_obs[0] (body_z)]  
  risks: 过度奖励高度可能导致机器人专注于跳跃或维持特定高度而放弃前进

- role_id: upright_orientation  
  purpose: 惩罚身体倾斜，促使保持竖直姿态  
  condition_to_use: 通过 body_up_z 偏离 1 的程度给予平方/线性惩罚，通常全程使用  
  usable_signals: [next_obs[1:5] 计算 body_up_z]  
  risks: 过于强制可能抑制正常的步态微调，导致僵硬动作

- role_id: action_energy_penalty / smoothness  
  purpose: 惩罚过大的扭矩输出，降低能耗并鼓励平滑控制  
  condition_to_use: 始终可用，将其作为小权重正则项  
  usable_signals: [action (8维扭矩)]  
  risks: 权重过大会抑制探索，使机器人无法生成有力步伐，导致前进速度降低

- role_id: lateral_drift_penalty  
  purpose: 惩罚侧向（y）速度，抑制横向漂移  
  condition_to_use: 前进为主目标时，通常作为辅助项使用  
  usable_signals: [next_obs[14] (body_y_velocity)]  
  risks: 轻微漂移可能无害，强制为 0 可能干扰转弯（本任务不要求转弯，可接受）

- role_id: vertical_oscillation_penalty  
  purpose: 惩罚垂直速度过大，减少上下颠簸  
  condition_to_use: 可与高度保持信号共存，防止剧烈跳跃  
  usable_signals: [next_obs[15] (body_z_velocity)]  
  risks: 可能抑制正常的步态引起的轻微起伏

### 10.3 慎用/禁用职责 avoid_roles
- role_id: distance_from_start / whole_trajectory_progress  
  reason: 禁止获取 x/y 绝对坐标，无法计算累积位移，且官方定位信息被屏蔽  
  forbidden_or_missing_signals: [x_position, y_position, distance_from_origin] 均在 forbidden_info_fields 中

- role_id: contact_consistency_reward  
  reason: 本版本无接触力信息，接触模式不可知  
  forbidden_or_missing_signals: [contact forces, foot contact states] 不在 obs 空间

- role_id: any_official_reward_reproduce  
  reason: 官方奖励被掩码且明确禁止使用或重构  
  forbidden_or_missing_signals: [reward_forward, reward_ctrl, etc.] 都在 forbidden_info_fields

- role_id: goal_reaching / sparse_event  
  reason: 任务无明确目标位置，非导航/稀疏探索类型  
  reason_detail: 终止仅基于高度超限或数值问题，无成功触发标志

## 11. role_to_signal_mapping
| role_id                      | usable signals                                      | missing signals      | candidate formula operators              | notes                                                                             |
|------------------------------|-----------------------------------------------------|----------------------|------------------------------------------|-----------------------------------------------------------------------------------|
| forward_velocity_reward      | next_obs[13] (body_x_velocity)                     
