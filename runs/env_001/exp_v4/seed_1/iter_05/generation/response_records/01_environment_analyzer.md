# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个物体从视口顶部中央附近开始，受到初始随机力的作用。目标是尽可能快地到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求"到达并稳定在中央目标着陆平台"，核心目标是导航到目标位置并保持稳定，同时优化燃料消耗。这符合导航目标到达任务的定义，其中包含到达目标、速度控制、姿态稳定等子目标。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推断)
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
- 动作数量: 4
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 点火左侧姿态发动机
- action 2: main_engine - 点火主发动机
- action 3: right_orientation_engine - 点火右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 当机体停止运动并稳定时，可能表示成功着陆
- failure-like termination: crash_or_body_contact - 坠毁或机体接触（非支撑接触），表示失败
- failure-like termination: horizontal_position_outside_viewport - 水平位置超出视口，表示偏离目标区域
- ambiguous termination: 无
- truncation: False (step返回的truncated固定为False)

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false - 没有明确的成功标志
- explicit_failure_flag_available: false - 没有明确的失败标志
- allowed_info_fields: {} - step返回的info为空字典，没有额外信息字段
- forbidden_or_uncertain_info_fields: 所有info字段均不可用，因为info固定为{}

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs - 当前状态的观测值
- action - 当前执行的动作
- next_obs - 执行动作后的下一状态观测值
- info - 当前为{}，无可用字段
- training_progress - 仅当prompt明确允许时才使用

禁止使用：
- original_reward - 官方奖励已被屏蔽，禁止使用
- official_reward - 禁止使用
- 未声明的info字段 - info固定为空字典
- 未声明的obs切片 - 仅允许使用obs[0]到obs[7]的明确定义字段

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) - 相对于目标的位置
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) - 线速度
- orientation: obs[4] (body_angle) - 姿态角
- angular_velocity: obs[5] (angular_velocity) - 角速度
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) - 支撑接触标志
- action: action (0-3) - 当前执行的动作，可用于惩罚燃料消耗

## 8. 不确定或不可用的信号
- 终止条件的具体阈值：不知道crash/outside_viewport/settled的具体判定阈值
- 目标平台的精确位置：只知道obs是相对坐标，但不知道目标平台的具体尺寸
- 初始随机力的分布：不知道初始随机力的具体参数
- 物理引擎参数：不知道重力、摩擦力、引擎推力等物理参数
- 时间步长：不知道每个step对应的时间增量
- 成功着陆的判定标准：不知道"settled"的具体判定条件（速度阈值、角度阈值等）
- 燃料消耗量：没有直接的燃料消耗观测值，只能通过动作类型间接推断
- 接触力的具体数值：只有接触标志（0/1），没有接触力大小信息
