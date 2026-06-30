# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个物体从视口顶部中央附近开始，受到初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求"到达并稳定在中央目标着陆平台"，核心目标是导航到目标位置并稳定着陆，同时优化燃料消耗。这符合导航到达任务的特征，而非单纯的平衡、连续控制或抓取任务。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推测)
- obs[0]: x_position - 相对于目标着陆平台的水平坐标
- obs[1]: y_position - 相对于着陆平台高度的垂直坐标
- obs[2]: x_velocity - 水平线速度
- obs[3]: y_velocity - 垂直线速度
- obs[4]: body_angle - 机体姿态角
- obs[5]: angular_velocity - 角速度
- obs[6]: left_support_contact - 左侧支撑接触标志 (1.0 表示接触, 0.0 表示未接触)
- obs[7]: right_support_contact - 右侧支撑接触标志 (1.0 表示接触, 0.0 表示未接触)

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 点火左侧姿态发动机
- action 2: main_engine - 点火主发动机
- action 3: right_orientation_engine - 点火右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 当物体不再运动或已稳定着陆时终止，这可能是成功着陆的标志
- failure-like termination: crash_or_body_contact - 发生碰撞或非预期身体接触时终止，这代表失败
- failure-like termination: horizontal_position_outside_viewport - 水平位置超出视口范围，代表失控或偏离目标
- ambiguous termination: 无明确区分成功/失败的单一终止标志
- truncation: False (step返回的truncated始终为False)

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false - 没有独立的成功标志
- explicit_failure_flag_available: false - 没有独立的失败标志
- allowed_info_fields: {} - step返回的info为空字典，无额外信息可用
- forbidden_or_uncertain_info_fields: 所有info字段均不可用，因为info始终为空

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs - 当前观测值
- action - 当前执行的动作
- next_obs - 执行动作后的下一观测值
- info - 当前为空字典，无可用字段
- training_progress - 仅当prompt明确允许时才使用

禁止使用：
- original_reward - 官方奖励已被屏蔽，禁止使用
- official_reward - 禁止使用
- 未声明的info字段 - info为空，无可用字段
- 未声明的obs切片 - 仅使用上述8个已定义字段

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) - 相对于目标的位置
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) - 线速度
- orientation: obs[4] (body_angle) - 姿态角
- angular_velocity: obs[5] (angular_velocity) - 角速度
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) - 接触标志
- action/engine: action (0-3) - 动作选择，可用于惩罚燃料消耗

## 8. 不确定或不可用的信号
- 终止原因区分：无法从step返回中区分成功着陆(crash_or_body_contact)与失败碰撞(body_not_awake_or_settled)，因为terminated是三个条件的逻辑或
- 目标平台精确位置：obs[0]和obs[1]是相对坐标，但不知道目标平台的具体尺寸和允许的着陆误差范围
- 燃料/能量消耗：没有直接的燃料剩余或消耗量观测值，只能通过动作类型间接推断
- 时间步数：没有时间步计数器，无法直接惩罚时间消耗
- 着陆质量：没有着陆速度、冲击力等质量指标
- 初始随机力：没有关于初始随机力的信息，无法补偿或利用
- 视口边界：没有视口边界的具体数值，无法判断距离边界的余量