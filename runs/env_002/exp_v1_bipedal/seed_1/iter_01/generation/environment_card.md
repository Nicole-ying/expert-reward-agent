# Env_002 环境理解卡片

## 1. 任务目标
这是一个2D双足行走任务。智能体需要控制一个双足机器人在不平坦地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。智能体必须协调两条腿的髋关节和膝关节，产生稳定的双足步态。如果身体摔倒则终止回合。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 任务描述明确为"2D locomotion task"，要求双足机器人行走，动作空间为连续关节力矩控制，观察空间包含运动学状态和地形感知信息，属于典型的连续控制运动任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 (推测)
- obs[0]: hull_angle - 主体相对于竖直方向的角度
- obs[1]: hull_angular_velocity - 主体的角速度
- obs[2]: horizontal_velocity - 前后线性速度
- obs[3]: vertical_velocity - 上下线性速度
- obs[4]: hip1_angle - 腿1髋关节角度
- obs[5]: hip1_speed - 腿1髋关节角速度
- obs[6]: knee1_angle - 腿1膝关节角度
- obs[7]: knee1_speed - 腿1膝关节角速度
- obs[8]: hip2_angle - 腿2髋关节角度
- obs[9]: hip2_speed - 腿2髋关节角速度
- obs[10]: knee2_angle - 腿2膝关节角度
- obs[11]: knee2_speed - 腿2膝关节角速度
- obs[12]: leg1_contact - 腿1地面接触标志 (1.0=接触, 0.0=未接触)
- obs[13]: leg2_contact - 腿2地面接触标志 (1.0=接触, 0.0=未接触)
- obs[14]: lidar_0 - LIDAR测距仪测量值0
- obs[15]: lidar_1 - LIDAR测距仪测量值1
- obs[16]: lidar_2 - LIDAR测距仪测量值2
- obs[17]: lidar_3 - LIDAR测距仪测量值3
- obs[18]: lidar_4 - LIDAR测距仪测量值4
- obs[19]: lidar_5 - LIDAR测距仪测量值5
- obs[20]: lidar_6 - LIDAR测距仪测量值6
- obs[21]: lidar_7 - LIDAR测距仪测量值7
- obs[22]: lidar_8 - LIDAR测距仪测量值8
- obs[23]: lidar_9 - LIDAR测距仪测量值9

## 4. 动作空间 action_space
- type: Box (连续)
- shape: (4,)
- bounds: [-1.0, 1.0] 每个关节
- action 0: hip_torque_leg1 - 施加在腿1髋关节的力矩
- action 1: knee_torque_leg1 - 施加在腿1膝关节的力矩
- action 2: hip_torque_leg2 - 施加在腿2髋关节的力矩
- action 3: knee_torque_leg2 - 施加在腿2膝关节的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain - 到达地形终点，可视为成功完成行走任务
- failure-like termination: body_fallen_over - 身体摔倒，可视为失败
- ambiguous termination: 无
- truncation: 无（step返回的truncated始终为False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false - step返回的info为空字典{}，没有显式成功标志
- explicit_failure_flag_available: false - step返回的info为空字典{}，没有显式失败标志
- allowed_info_fields: 无（info始终为空字典{}）
- forbidden_or_uncertain_info_fields: 所有info字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs - 当前步的观测（24维向量）
- action - 当前步执行的动作（4维向量）
- next_obs - 下一步的观测（24维向量）
- info - 当前步的info字典（但此环境中info始终为空{}，无可用字段）
- training_progress - 仅当prompt明确允许时才使用

禁止使用：
- original_reward - 官方奖励已被屏蔽，禁止使用
- official_reward - 禁止使用
- 未声明的info字段 - info始终为空
- 未声明的obs切片 - 只能使用已声明的obs[0]到obs[23]

## 7. 可用于奖励函数的信号
- position: 可通过obs[2]水平速度积分估算前进距离；obs[3]垂直速度
- velocity: obs[2]水平速度（前进速度），obs[3]垂直速度，obs[1]角速度
- orientation: obs[0]主体角度（保持直立的关键信号）
- contact: obs[12]腿1接触，obs[13]腿2接触（步态相位检测）
- action/engine: action[0-3]关节力矩（可用于能量消耗惩罚）
- joint state: obs[4-11]关节角度和速度（步态协调性）
- terrain: obs[14-23]LIDAR测量值（地形感知）

## 8. 不确定或不可用的信号
- 绝对位置坐标：观测中没有x/y位置信息，只有速度
- 地形终点距离：没有显式给出到终点的距离
- 能量消耗：没有直接的能耗测量，只能通过action力矩间接估计
- 步态周期/相位：没有显式步态相位信息，需从接触信号推断
- 摔倒检测：terminated=True且body_fallen_over为真时才知道摔倒，但无法从观测中直接判断是否即将摔倒
- 成功标志：没有显式success标志，只能通过terminated=True且reached_end_of_terrain为真推断成功
- 失败标志：没有显式failure标志，只能通过terminated=True且body_fallen_over为真推断失败
- info字典：始终为空{}，无任何额外信号可用