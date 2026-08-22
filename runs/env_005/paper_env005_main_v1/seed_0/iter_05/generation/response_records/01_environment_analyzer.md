# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个连续控制运动任务。一个三维四足机器人具有四条腿和八个力矩控制关节，必须持续向前行走或奔跑，同时保持身体直立。机器人的身体高度必须维持在健康范围（0.2 < z < 1.0）之内；若身体高度低于 0.2（倒塌）或高于 1.0（弹跳出范围）或任何状态值变为非有限，则回合提前终止。策略应实现稳定且持续的前向运动，而非仅仅保持站立。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box（连续值）
- shape: (105,)
- dtype: float32（推断，无明确说明）
- obs[0]：主体垂直位置 body_z
- obs[1:5]：主体方向四元数（body_orientation_1 ~ body_orientation_4，4 个分量）
- obs[5:13]：8 个关节角度（joint_1_angle ~ joint_8_angle）
- obs[13]：世界坐标系前向速度 body_x_velocity
- obs[14]：世界坐标系横向速度 body_y_velocity
- obs[15]：垂直速度 body_z_velocity
- obs[16]：滚动角速度 body_roll_velocity
- obs[17]：俯仰角速度 body_pitch_velocity
- obs[18]：偏航角速度 body_yaw_velocity
- obs[19:27]：8 个关节角速度（joint_1_velocity ~ joint_8_velocity）
- obs[27:105]：外部接触力与力矩（78 维）。按 13 个身体部件（躯干 + 4 条腿的上段、下段、足段）各 6 维 [fx, fy, fz, tx, ty, tz] 排列。足部接触信号较强的索引：
  - obs[45:51]：前左脚（front_left foot）
  - obs[63:69]：前右脚（front_right foot）
  - obs[81:87]：后左脚（back_left foot）
  - obs[99:105]：后右脚（back_right foot）

## 4. 动作空间 action_space
- type: Box（连续，力矩归一化到 [-1, 1]）
- shape: (8,)
- 每个动作对应一个关节的力矩指令：
  - action 0: hip_1（前左髋关节力矩）
  - action 1: ankle_1（前左踝关节力矩）
  - action 2: hip_2（前右髋关节力矩）
  - action 3: ankle_2（前右踝关节力矩）
  - action 4: hip_3（后左髋关节力矩）
  - action 5: ankle_3（后左踝关节力矩）
  - action 6: hip_4（后右髋关节力矩）
  - action 7: ankle_4（后右踝关节力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无（不存在明确的成功触发条件）
- failure-like termination: 身体高度超出 (0.2, 1.0) 范围或任何状态值为 NaN/Inf
- ambiguous termination: 无
- truncation: 达到内部时间限制（time_limit_reached）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典，不可用）
- forbidden_or_uncertain_info_fields: 所有字段均被禁止或不存在，特别是 reward_forward、reward_ctrl、reward_contact、reward_survive、x_position、y_position、distance_from_origin、x_velocity、y_velocity 等均不可用。

*注：由于 info 为空，终止原因不能直接从标志位获取。然而，reward 函数可以通过 next_obs[0]（body_z）是否在 [0.2, 1.0] 之外来推断是否发生失败式终止；同样，若 next_obs 包含 NaN 或 Inf，则可能需要防御性处理。*

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前状态向量 (105,)
- action: 当前动作向量 (8,)
- next_obs: 下一状态向量 (105,)
- info 中明确允许的字段：无（所有字段被禁止）
- training_progress：仅当 prompt 明确允许时使用（本环境 prompt 未允许，故不应使用）

禁止使用：
- original_reward（已被遮蔽，不可用）
- official_reward 或其任何成分
- 未声明的 info 字段或 obs 切片
- 任何全局位置信息（x/y 坐标）或全局速度（x/y 速度）除非在 obs 中显式给出

## 7. 可用于奖励函数的信号
- position: body_z (obs[0])、主体姿态四元数 (obs[1:5])、关节角度 (obs[5:13])
- velocity: body_x_velocity (obs[13])、body_y_velocity (obs[14])、body_z_velocity (obs[15])、角速度 (obs[16:18])、关节角速度 (obs[19:27])
- orientation: 主体方向四元数（可转换为倾斜程度）
- contact: 各身体部件的接触力/力矩 (obs[27:105])，尤其是四个足部的接触信号
- action/engine: 动作本身 (action, 8维) 可用于惩罚过大力矩或抖动

## 8. 不确定或不可用的信号
- 官方奖励项（reward_forward、reward_ctrl 等）被强制屏蔽，不可使用
- 全局绝对位置（x_position, y_position, distance_from_origin）不在观察中且 info 禁止
- 速度字段 x_velocity、y_velocity 在 info 中禁止，但 body_x_velocity (obs[13]) 仍可用
- 明确的成功/失败标志不存在
- 剩余步数或时间信息不可直接获得
- training_progress 是否可用不确定（未明确说明，建议不使用）
- 终止原因标签（如 fell_over、height_fault 等）不存在
