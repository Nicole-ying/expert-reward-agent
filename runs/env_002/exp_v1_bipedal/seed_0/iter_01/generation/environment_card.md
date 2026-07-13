# Env_002 环境理解卡片

## 1. 任务目标
这是一个2D双足行走任务。智能体需要控制一个双足机器人在不平坦地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。智能体必须协调两条腿的髋关节和膝关节，产生稳定的双足步态。如果身体摔倒则终止回合。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 任务描述明确为"2D locomotion task"，目标是行走前进，动作空间是连续关节力矩控制，观察空间包含身体姿态、关节角度、速度等典型运动控制特征，属于典型的连续控制运动任务。

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
- obs[12]: leg1_contact - 腿1地面接触标志（1.0=接触，0.0=无接触）
- obs[13]: leg2_contact - 腿2地面接触标志（1.0=接触，0.0=无接触）
- obs[14..23]: lidar_0..lidar_9 - 10个LIDAR测距仪沿前方地形的距离测量值

## 4. 动作空间 action_space
- type: Box (连续)
- shape: (4,)
- bounds: [-1.0, 1.0] 每个关节
- action 0: hip_torque_leg1 - 施加在腿1髋关节上的力矩
- action 1: knee_torque_leg1 - 施加在腿1膝关节上的力矩
- action 2: hip_torque_leg2 - 施加在腿2髋关节上的力矩
- action 3: knee_torque_leg2 - 施加在腿2膝关节上的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain - 到达地形终点，可视为成功完成
- failure-like termination: body_fallen_over - 身体摔倒，可视为失败
- ambiguous termination: 无
- truncation: 无（step返回的truncated始终为False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info字典为空{}，没有显式success标志）
- explicit_failure_flag_available: false（info字典为空{}，没有显式failure标志）
- allowed_info_fields: 无（info始终为{}）
- forbidden_or_uncertain_info_fields: 所有info字段（因为info为空）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs - 当前步的观察（24维向量）
- action - 当前步执行的动作（4维向量）
- next_obs - 下一步的观察（24维向量）
- info - 当前步的info字典（但此环境中info始终为空{}）
- training_progress - 仅当prompt明确允许时才使用

禁止使用：
- original_reward - 官方奖励已被屏蔽，禁止使用
- official_reward - 禁止使用
- 未声明的info字段 - info始终为空
- 未声明的obs切片 - 只能使用上述定义的24维观察

## 7. 可用于奖励函数的信号
- position: 可通过水平速度(obs[2])积分或LIDAR(obs[14..23])间接推断前进距离
- velocity: 水平速度(obs[2])、垂直速度(obs[3])、关节角速度(obs[5],7,9,11)
- orientation: 主体角度(obs[0])、主体角速度(obs[1])
- contact: 腿1接触(obs[12])、腿2接触(obs[13])
- action/engine: 动作(action[0..3])可用于计算能量消耗

## 8. 不确定或不可用的信号
- 地形高度/不平坦程度：LIDAR数据(obs[14..23])可能包含地形信息，但具体含义不确定
- 能量消耗：没有直接的能耗测量，只能通过动作(action)的平方和近似
- 步态周期/相位：没有显式的步态相位信息
- 到达终点的距离：没有直接的距离测量，只能通过LIDAR或速度推断
- 摔倒检测：只能通过终止条件间接知道，但终止时无法获取next_obs
- 关节位置限制：没有关节角度限制信息
- 地面摩擦力/地形特性：没有显式测量