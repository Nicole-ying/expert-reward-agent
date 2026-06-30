# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器轨迹优化任务。一个飞行器从视口顶部中央附近开始，受到初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求"到达并稳定在中央目标平台"，核心目标是导航到目标位置并保持稳定，同时优化燃料消耗。这符合导航到达任务的定义。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推断)
- obs[0]: x_position - 相对于目标平台的水平坐标
- obs[1]: y_position - 相对于平台高度的垂直坐标
- obs[2]: x_velocity - 水平线速度
- obs[3]: y_velocity - 垂直线速度
- obs[4]: body_angle - 机体角度（姿态角）
- obs[5]: angular_velocity - 角速度
- obs[6]: left_support_contact - 左侧支撑接触标志（1.0=接触，0.0=未接触）
- obs[7]: right_support_contact - 右侧支撑接触标志（1.0=接触，0.0=未接触）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 点火左侧姿态引擎
- action 2: main_engine - 点火主引擎
- action 3: right_orientation_engine - 点火右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 机体停止运动/稳定在平台上，可能是成功着陆的标志
- failure-like termination: crash_or_body_contact - 坠毁或非正常机体接触；horizontal_position_outside_viewport - 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 无（step返回truncated=False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info为空字典，无显式成功标志）
- explicit_failure_flag_available: false（info为空字典，无显式失败标志）
- allowed_info_fields: 无（info始终返回空字典{}）
- forbidden_or_uncertain_info_fields: 所有info字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前状态）
- action（当前动作）
- next_obs（下一状态）
- training_progress（仅当prompt明确允许时才使用）

禁止使用：
- original_reward（官方奖励已屏蔽）
- info（始终为空字典）
- 未声明的obs切片

## 7. 可用于奖励函数的信号
- position: obs[0] x_position, obs[1] y_position - 相对于目标的位置
- velocity: obs[2] x_velocity, obs[3] y_velocity - 线速度
- orientation: obs[4] body_angle - 姿态角
- angular_velocity: obs[5] angular_velocity - 角速度
- contact: obs[6] left_support_contact, obs[7] right_support_contact - 支撑接触标志
- action: action（0-3） - 当前执行的动作，可用于惩罚引擎使用

## 8. 不确定或不可用的信号
- 终止原因：无法区分成功（稳定着陆）和失败（坠毁/出界），因为terminated是三个条件的逻辑或
- 燃料/能量消耗：未在观测空间中提供
- 时间步数：未在观测空间中提供
- 目标距离：需要从obs[0]和obs[1]计算
- 接触力大小：只有接触标志，没有力的大小信息
- 引擎推力大小：只有动作类型，没有推力幅度信息