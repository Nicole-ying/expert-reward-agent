# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 双足行走任务。智能体控制一个拥有两条腿（各含髋、膝两个关节）的身体，在不平坦地形上向前行走。核心目标是 **走得尽可能远、尽可能快**，同时 **最小化能量消耗**。要获得好成绩，必须通过协调四肢关节力矩生成稳定、高效的双足步态；一旦身体倾斜过度、摔倒，回合即告失败。任务**不存在明确的到达点**，过程终止于身体摔倒（失败）或到达地形末端（可视为成功完成路程）。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务明确要求持续前进通过地形，核心判断依据是推进距离与速度，而非导航到固定目标点；能耗作为次要优化项，不构成新的主任务类型。

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float（具体为 float32/64，下同）
- 各分量含义及 reward_usable 标记：

| index | name | meaning | reward_usable |
|-------|------|---------|---------------|
| 0 | hull_angle | 躯干相对于竖直方向的倾角（弧度） | true |
| 1 | hull_angular_velocity | 躯干的角速度 | true |
| 2 | horizontal_velocity | 前方的线速度（正值表示向前） | true |
| 3 | vertical_velocity | 上下方向线速度 | true |
| 4 | hip1_angle | 腿1髋关节角度 | true |
| 5 | hip1_speed | 腿1髋关节角速度 | true |
| 6 | knee1_angle | 腿1膝关节角度 | true |
| 7 | knee1_speed | 腿1膝关节角速度 | true |
| 8 | leg1_contact | 腿1触地标志（1.0 触地，0.0 离地） | true |
| 9 | hip2_angle | 腿2髋关节角度 | true |
|10 | hip2_speed | 腿2髋关节角速度 | true |
|11 | knee2_angle | 腿2膝关节角度 | true |
|12 | knee2_speed | 腿2膝关节角速度 | true |
|13 | leg2_contact | 腿2触地标志（1.0 触地，0.0 离地） | true |
|14 | lidar_0 | LIDAR 距离传感器 0（前方地形距离） | true（但需谨慎使用） |
|15 | lidar_1 | LIDAR 距离传感器 1 | true |
|16 | lidar_2 | LIDAR 距离传感器 2 | true |
|17 | lidar_3 | LIDAR 距离传感器 3 | true |
|18 | lidar_4 | LIDAR 距离传感器 4 | true |
|19 | lidar_5 | LIDAR 距离传感器 5 | true |
|20 | lidar_6 | LIDAR 距离传感器 6 | true |
|21 | lidar_7 | LIDAR 距离传感器 7 | true |
|22 | lidar_8 | LIDAR 距离传感器 8 | true |
|23 | lidar_9 | LIDAR 距离传感器 9 | true |

> 注：所有字段在理论上都可参与奖励计算。LIDAR 读数可用于感知前方地形起伏，但作为连续控制任务，直接奖励地形平整度可能引入噪声；除非明确需要地形自适应，否则奖励函数不一定要使用它们。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- dtype: float  
- 各维度含义：

| index | name | meaning |
|-------|------|---------|
| 0 | hip_torque_leg1 | 施加到腿1髋关节的力矩，范围 [-1, 1] |
| 1 | knee_torque_leg1 | 施加到腿1膝关节的力矩，范围 [-1, 1] |
| 2 | hip_torque_leg2 | 施加到腿2髋关节的力矩，范围 [-1, 1] |
| 3 | knee_torque_leg2 | 施加到腿2膝关节的力矩，范围 [-1, 1] |

动作是连续的关节力矩，智能体需生成协调的关节动作以形成稳定步态。

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**：`reached_end_of_terrain` — 身体抵达地形终点，说明已走完全程。由于无明确成功标志，该终止仅暗示路径被完整覆盖，必须谨慎使用，不能直接作为奖励。
- **failure-like termination**：`body_fallen_over` — 躯干倾斜过度导致摔倒，这是明确的失败。
- **ambiguous termination**：无其它模式。
- **truncation**：源码中仅返回 `terminated`，无 `truncated` 标记，因此默认所有终止都视为 episode 结束（可能由最大步数截断，但未指明，需假设环境可能包含基于步数的截断）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
  （无 `info` 输出，无法直接得知 `reached_end_of_terrain` 的具体值）
- explicit_failure_flag_available: **false**  
  （同样，`body_fallen_over` 没有作为独立标志提供给 reward 函数）
- allowed_info_fields: 无（`info` 为 `{}`）
- forbidden_or_uncertain_info_fields: 任何需要从 `info` 读取的字段均禁止

> 奖励函数**必须**基于观测序列隐式推断失败风险（例如通过 `hull_angle` 绝对值过大），**绝对不能**依赖 `done` 或任何终止标志。

## 6. reward 函数接口契约

函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

- **允许使用**：
  - `obs`（当前观测数组，shape (24,)）
  - `action`（当前动作数组，shape (4,)）
  - `next_obs`（下一时刻观测数组）
  - `training_progress`（浮点数，范围 [0,1]，仅在 prompt 明确允许时使用）
- **禁止使用**：
  - `original_reward`（被屏蔽的官方奖励）
  - 任何未声明的 `info` 字段（本环境中 `info` 为空，故禁止）
  - 未出现在上述列表中的额外全局变量

## 7. 可用于奖励函数的信号

- **姿态/平衡**：
  - `hull_angle`（obs[0]）：躯干偏离竖直的程度，越小越平衡。
  - `hull_angular_velocity`（obs[1]）：倾斜变化快慢，可用于惩罚剧烈晃动。
- **推进速度**：
  - `horizontal_velocity`（obs[2]）：正向速度，越大前进越快。
  - `vertical_velocity`（obs[3]）：可辅助检测颠簸或跳跃，但不一定是主要奖励源。
- **关节状态**：
  - 髋/膝角度与角速度（obs[4..7], obs[9..12]）：可用于限制关节极限、惩罚过大动作或鼓励平滑运动。
- **接触状态**：
  - `leg1_contact`, `leg2_contact`（obs[8], [13]）：反映脚是否着地，可用于检测步态交替或防止双腿同时离地。
- **动作/能量**：
  - `action` 四维力矩：直接反映控制能量，平方和或绝对值和可用来惩罚低效动作。
- **地形感知**：
  - LIDAR 传感器（obs[14..23]）：10个距离读数，描述前方地形轮廓，可用于奖励对地形的适应性（如避免过陡坡），但使用需谨慎，会大幅增加奖励设计复杂度。

## 8. 不确定或不可用的信号

- **明确的成功/失败标志**：不可用，`info` 为空。
- **地形结束信号**：无法在奖励计算时获取。
- **地形确切形状**：LID