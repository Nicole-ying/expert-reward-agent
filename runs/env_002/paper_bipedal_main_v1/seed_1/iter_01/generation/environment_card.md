# 匿名环境理解卡片

## 1. 任务目标
控制一个双足机器人在不平坦的二维地形上行走。主要目标是让机器人尽可能远、尽可能快地向前移动，同时尽量降低能量消耗。机器人需要通过协调两条腿的髋关节和膝关节来产生稳定的步态。如果机器人身体倒地则立即终止，若能到达地形尽头也会终止。任务鼓励在到达终点前走得更远、更快，并保持低能耗与身体直立。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32
- obs[0]: hull_angle —— 身体相对于直立方向的角度 (弧度)
- obs[1]: hull_angular_velocity —— 身体的角速度
- obs[2]: horizontal_velocity —— 向前/向后线速度
- obs[3]: vertical_velocity —— 向上/向下线速度
- obs[4]: hip1_angle —— 腿1髋关节角度
- obs[5]: hip1_speed —— 腿1髋关节角速度
- obs[6]: knee1_angle —— 腿1膝关节角度
- obs[7]: knee1_speed —— 腿1膝关节角速度
- obs[8]: leg1_contact —— 腿1接触地面标志 (1.0=接触, 0.0=未接触)
- obs[9]: hip2_angle —— 腿2髋关节角度
- obs[10]: hip2_speed —— 腿2髋关节角速度
- obs[11]: knee2_angle —— 腿2膝关节角度
- obs[12]: knee2_speed —— 腿2膝关节角速度
- obs[13]: leg2_contact —— 腿2接触地面标志 (1.0=接触, 0.0=未接触)
- obs[14] ~ obs[23]: lidar_0 ~ lidar_9 —— 10个前方地形激光雷达测距值，表示到地形表面的距离或障碍物距离

## 4. 动作空间 action_space
- type: Box (连续)
- shape: (4,)
- bounds: 每个分量均在 [-1.0, 1.0]
- action 0: hip_torque_leg1 —— 施加到腿1髋关节的扭矩
- action 1: knee_torque_leg1 —— 施加到腿1膝关节的扭矩
- action 2: hip_torque_leg2 —— 施加到腿2髋关节的扭矩
- action 3: knee_torque_leg2 —— 施加到腿2膝关节的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形尽头，episode 正常结束，可视为成功）
- failure-like termination: body_fallen_over（身体倒地，走不下去）
- ambiguous termination: 无
- truncation: 无（本任务没有设置时间步上限的截断，终止只来自上述两个条件）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （info 为空字典，没有任何可用字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不提供，禁止依赖 info 做奖励判断

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs （当前观察）
- action （当前动作）
- next_obs （下一观察）
- info 中明确允许的字段：无（info 为空，不可用任何字段）
- training_progress：本任务 prompt 未明确允许使用，应避免依赖

禁止使用：
- original_reward （被屏蔽的官方奖励）
- official_reward 或任何以“official”命名的变量
- 未声明的 info 字段
- 未声明的 obs 切片或超出上述定义范围的维度

## 7. 可用于奖励函数的信号
- position / progress: 可通过 horizontal_velocity 累加推断水平移动距离（next_obs 与 obs 的速度积分），但没有直接的 x 坐标。
- velocity: horizontal_velocity（前进速度），vertical_velocity（用于惩罚不必要的跳起）。
- orientation: hull_angle（身体倾斜，倾向于惩罚偏离0），hull_angular_velocity（惩罚快速旋转）。
- contact: leg1_contact, leg2_contact（可用于步态对称性、防止同时离地等）。
- action / engine: 动作扭矩的平方和或绝对值可作为能耗惩罚。
- terrain sensing: lidar_0..9（可用来奖励对前方崎岖地形的预适应，但通常谨慎使用）。

## 8. 不确定或不可用的信号
- 没有显式的成功/失败标志（info 为空），不能直接用这些做奖励塑造。
- 没有直接的位置坐标（x, y），前进距离只能通过速度积分获取，可能受到积分误差影响。
- 没有地形的全局信息，只有局部雷达读数。
- 禁止使用 original_reward，因此不能通过监控其数值来获取基线。