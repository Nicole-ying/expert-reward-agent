# Response Record

# Env_002 环境理解卡片

## 1. 任务目标
这是一个2D双足行走任务。智能体需要控制一个双足机器人在不平坦地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。智能体必须协调两条腿的髋关节和膝关节，产生稳定的双足步态。如果身体摔倒则终止回合。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 任务描述明确为"2D locomotion task"，要求双足机器人行走，动作空间为连续关节力矩控制，观察空间包含身体姿态、关节角度、速度、地面接触和地形感知信息，属于典型的连续控制运动任务。

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
- obs[14..23]: lidar_0..9 - 10个LIDAR测距仪沿前方地形的距离测量值

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
- explicit_success_flag_available: false（info字典为空{}，无显式成功标志）
- explicit_failure_flag_available: false（info字典为空{}，无显式失败标志）
- allowed_info_fields: 无（info始终为{}）
- forbidden_or_uncertain_info_fields: 所有info字段（因为info为空字典）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前观测（24维向量）
- action: 当前动作（4维向量）
- next_obs: 下一时刻观测（24维向量）
- info: 当前为空字典{}，无可用字段
- training_progress: 仅当prompt明确允许时才使用

禁止使用：
- original_reward: 官方奖励已被屏蔽，禁止使用
- official_reward: 禁止使用
- 未声明的info字段：info始终为空
- 未声明的obs切片：仅允许使用上述定义的24个字段

## 7. 可用于奖励函数的信号
- position: 可通过水平速度(obs[2])积分或直接使用水平速度作为前进奖励
- velocity: 水平速度(obs[2])、垂直速度(obs[3])、角速度(obs[1])
- orientation: 主体角度(obs[0])，可用于奖励直立姿态
- contact: 腿1接触(obs[12])、腿2接触(obs[13])，可用于检测步态周期
- action/engine: 动作(action[0..3])，可用于计算能量消耗惩罚
- joint angles: 髋关节角度(obs[4], obs[8])、膝关节角度(obs[6], obs[10])
- joint speeds: 髋关节速度(obs[5], obs[9])、膝关节速度(obs[7], obs[11])
- terrain perception: LIDAR数据(obs[14..23])，可用于地形适应

## 8. 不确定或不可用的信号
- 终止原因：无法区分是摔倒还是到达终点导致的终止（terminated=True但info为空）
- 前进距离：没有直接的距离测量，只能通过速度间接估计
- 能量消耗：没有直接的能耗测量，只能通过动作幅度和关节速度近似
- 步态质量：没有直接的步态稳定性或对称性指标
- 地形高度：LIDAR提供距离测量，但无直接地形高度信息
- 时间步数：step函数未提供时间信息，需外部跟踪
- 摔倒检测阈值：不知道body_fallen_over的具体判定条件（角度阈值等）
