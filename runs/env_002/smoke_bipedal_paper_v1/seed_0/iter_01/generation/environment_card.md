# 匿名环境理解卡片

## 1. 任务目标
本环境是一个二维双足步行任务。
智能体需要控制两条腿的髋关节和膝关节，在崎岖不平的地形上**尽可能远、尽可能快地稳定行走**，同时尽量**减少能量消耗**。  
身体一旦**摔倒**，回合立即终止；若**到达地形终点**，也视为终止。  
核心要求是产生稳健的双足步态，保持前进。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（推测连续值）
- obs[0]: hull_angle — 身体相对于竖直方向的角度 (rad)
- obs[1]: hull_angular_velocity — 身体角速度 (rad/s)
- obs[2]: horizontal_velocity — 水平线速度（前进方向） (m/s)
- obs[3]: vertical_velocity — 竖直线速度 (m/s)
- obs[4]: hip1_angle — 腿1髋关节角度 (rad)
- obs[5]: hip1_speed — 腿1髋关节角速度 (rad/s)
- obs[6]: knee1_angle — 腿1膝关节角度 (rad)
- obs[7]: knee1_speed — 腿1膝关节角速度 (rad/s)
- obs[8]: leg1_contact — 腿1触地标志 (1.0 接触, 0.0 不接触)
- obs[9]: hip2_angle — 腿2髋关节角度 (rad)
- obs[10]: hip2_speed — 腿2髋关节角速度 (rad/s)
- obs[11]: knee2_angle — 腿2膝关节角度 (rad)
- obs[12]: knee2_speed — 腿2膝关节角速度 (rad/s)
- obs[13]: leg2_contact — 腿2触地标志 (1.0 接触, 0.0 不接触)
- obs[14..23]: lidar_0..lidar_9 — 10 个激光测距值，测量前方地形距离 (m)

## 4. 动作空间 action_space
- type: Box (continuous)
- shape: [4]
- bounds: [-1.0, 1.0] 每个关节
- action 0: hip_torque_leg1 — 腿1髋关节力矩
- action 1: knee_torque_leg1 — 腿1膝关节力矩
- action 2: hip_torque_leg2 — 腿2髋关节力矩
- action 3: knee_torque_leg2 — 腿2膝关节力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `reached_end_of_terrain`（到达地形终点，可视为成功完成路线）
- failure-like termination: `body_fallen_over`（身体摔倒，失败）
- ambiguous termination: 无
- truncation: 未提及超时截断，推测由 `terminated` 直接控制，无单独 `truncation`

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 任何其他 info 字段（包括从未声明的 success、failure、termination_reason 等）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：当前观察（24维数组）
- `action`：当前动作（4维数组）
- `next_obs`：下一观察（24维数组）
- `info` 中明确允许的字段：无（info 为空，不能使用任何键）
- `training_progress`：**不允许使用**（任务描述未提及）

禁止使用：
- `original_reward`（官方奖励被屏蔽）
- 任何其他未声明的 `info` 字段
- 未在观察空间中声明的 obs 切片维度

## 7. 可用于奖励函数的信号
从 `obs` 或 `next_obs` 中可以提取以下信息构造自定义奖励：
- **水平速度**：`obs[2]` / `next_obs[2]`，用于鼓励快速前进。
- **身体倾斜（稳定性）**：`obs[0]`（角度）、`obs[1]`（角速度），用于处罚偏离直立。
- **垂直速度**：`obs[3]`，可能用于处罚过大的上下跳动。
- **能量消耗**：通过 `action` 的平方和或绝对值之和近似力矩能耗，鼓励动作平滑与节能。
- **接触模式**：`obs[8]`、`obs[13]` 触地标志，可用于塑造步态（例如处罚双脚同时离地或同时着地）。
- **关节状态**：`obs[4:8]` 和 `obs[9:13]` 的角度、角速度，用于限制关节极限或鼓励特定姿态。
- **前进距离**：通过对水平速度积分或在自定义代码中累加（需跨时间步累加，不能仅从 obs 直接得到绝对位置，但奖励设计时不能用“官方位置”之外的信息）。

## 8. 不确定或不可用的信号
- **绝对位置坐标**：观察空间内无 x/y 坐标，无法直接获取已行走距离或剩余距离。
- **地形终点距离**：激光雷达可能隐含前方信息，但没有明确的“剩余距离”字段。
- **成功/失败标志**：`info` 中无任何字段，不能基于终止原因奖励。
- **能耗真实值**：环境中没有提供能量消耗的测量值，只能用动作大小作为代理。
- **地形类型或难度**：完全匿名，未给出任何地形参数。
- **训练进度**：`training_progress` 不可用，因为任务描述未允许使用它。