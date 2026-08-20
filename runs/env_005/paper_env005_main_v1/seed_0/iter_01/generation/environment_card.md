# 匿名环境理解卡片

## 1. 任务目标
控制一个 3D 四足机器人，通过 8 个扭矩控制关节，实现持续、稳定的前向行走或奔跑，同时保持身体直立。机器人身体高度必须保持在健康范围内（0.2~1.0），过低（倒塌）或过高（弹飞）都会导致回合终止。核心目标是维持稳定的前向运动，而不是仅仅保持站立。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: [105]
- dtype: float
- obs[0]: body_z – 主体垂直位置
- obs[1:5]: body_orientation（四元数，共 4 维）– 主体姿态
- obs[5:13]: joint_angles（共 8 维）– 8 个关节角度
- obs[13]: body_x_velocity – 主体世界 x 轴速度（前向速度）
- obs[14]: body_y_velocity – 主体世界 y 轴速度（横向速度）
- obs[15]: body_z_velocity – 主体垂直速度
- obs[16]: body_roll_velocity – 滚转角速度
- obs[17]: body_pitch_velocity – 俯仰角速度
- obs[18]: body_yaw_velocity – 偏航角速度
- obs[19:27]: joint_velocities（共 8 维）– 8 个关节角速度
- obs[27:33]: torso 接触力/力矩（fx, fy, fz, tx, ty, tz）
- obs[33:39]: front_left upper leg 接触力/力矩
- obs[39:45]: front_left lower leg 接触力/力矩
- obs[45:51]: front_left foot 接触力/力矩
- obs[51:57]: front_right upper leg 接触力/力矩
- obs[57:63]: front_right lower leg 接触力/力矩
- obs[63:69]: front_right foot 接触力/力矩
- obs[69:75]: back_left upper leg 接触力/力矩
- obs[75:81]: back_left lower leg 接触力/力矩
- obs[81:87]: back_left foot 接触力/力矩
- obs[87:93]: back_right upper leg 接触力/力矩
- obs[93:99]: back_right lower leg 接触力/力矩
- obs[99:105]: back_right foot 接触力/力矩

## 4. 动作空间 action_space
- type: Box (continuous)
- shape: [8]
- 范围: [-1.0, 1.0]
- action 0: hip_1_torque – 第一个髋关节扭矩
- action 1: ankle_1_torque – 第一个踝关节扭矩
- action 2: hip_2_torque – 第二个髋关节扭矩
- action 3: ankle_2_torque – 第二个踝关节扭矩
- action 4: hip_3_torque – 第三个髋关节扭矩
- action 5: ankle_3_torque – 第三个踝关节扭矩
- action 6: hip_4_torque – 第四个髋关节扭矩
- action 7: ankle_4_torque – 第四个踝关节扭矩

（注：四足共 8 个驱动关节，对称分布；每腿由髋 + 踝关节组成）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式的成功终止条件；任务要求持续前进，只有时间截断（truncation）可能对应完成整个 episode
- failure-like termination:
  - body_height_outside_healthy_range（主体 z 超出 0.2~1.0）：倒塌或弹飞
  - state_value_outside_finite_range（任何状态量为 NaN 或 inf）
- ambiguous termination: 无
- truncation: time_limit_reached（达到最大步数限制）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []  （环境 step 返回的 info 为空 dict，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin, x_velocity, y_velocity 等均被禁止

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观察）
- action（当前动作）
- next_obs（下一观察）
- 不依赖 info（info 为空且不允许使用其字段）
- training_progress 仅当 prompt 明确允许时才用，本环境无此类指示，默认不使用

禁止使用：
- original_reward（被屏蔽）
- 任何 info 字段
- 任何未在 observation_space 中声明的数据

## 7. 可用于奖励函数的信号
- position:
  - body_z（obs[0]）– 垂直高度，可用于惩罚离健康范围偏差
- velocity:
  - body_x_velocity（obs[13]）– 前向速度，可直接作为前进奖励
  - body_y_velocity（obs[14]）– 横向速度，可用于抑制侧向漂移
  - body_z_velocity（obs[15]）– 垂直速度，结合高度变化可判断坠落倾向
- orientation:
  - body_orientation 四元数（obs[1:5]）– 可用于保持身体水平，惩罚倾斜
  - roll/pitch/yaw 角速度（obs[16:18]）– 可惩罚快速旋转，维持稳定姿态
- contact:
  - 各腿足部接触力（obs[45:51], [63:69], [81:87], [99:105]）– 判断地面接触，可用于奖励足部触地或惩罚不稳定的接触模式
- action/engine:
  - 关节扭矩 action[0:7] – 可用于惩罚过大扭矩，提高能效
  - 关节速度（obs[19:27]）– 可结合扭矩计算功率

## 8. 不确定或不可用的信号
- 官方奖励项（reward_forward, reward_ctrl 等）– 即使存在 info 中也被强制禁止
- 绝对平面位置（x_position, y_position）– 不可用
- 显式终止原因字符串（如 “termination_reason”）– 不存在
- 成功/失败布尔标志 – 不存在
- 任何需要跨步累积的量（如 episode 累计距离）– 奖励函数是无状态的，不可直接累积，但可以利用当前步的前向速度近似瞬时进步。