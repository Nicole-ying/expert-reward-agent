# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器轨迹优化任务。一个飞行器从视口顶部中央附近开始，受到初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求"到达并稳定在中央目标平台"，核心目标是导航到目标位置并保持稳定，同时优化燃料消耗，符合导航到达类任务的特征。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推测)
- obs[0]: x_position - 相对于目标平台的水平坐标
- obs[1]: y_position - 相对于平台高度的垂直坐标
- obs[2]: x_velocity - 水平线速度
- obs[3]: y_velocity - 垂直线速度
- obs[4]: body_angle - 机体姿态角
- obs[5]: angular_velocity - 角速度
- obs[6]: left_support_contact - 左侧支撑接触标志 (0.0 或 1.0)
- obs[7]: right_support_contact - 右侧支撑接触标志 (0.0 或 1.0)

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 启动左侧姿态引擎
- action 2: main_engine - 启动主引擎
- action 3: right_orientation_engine - 启动右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 机体停止运动或稳定在平台上，可能是成功着陆的标志
- failure-like termination: crash_or_body_contact - 坠毁或非正常机体接触；horizontal_position_outside_viewport - 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 无 (step返回的truncated始终为False)

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false (info字典为空，无显式成功标志)
- explicit_failure_flag_available: false (info字典为空，无显式失败标志)
- allowed_info_fields: 无 (info始终为{})
- forbidden_or_uncertain_info_fields: 所有info字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs - 当前观测
- action - 当前动作
- next_obs - 下一步观测
- info - 当前为空字典，无可用字段
- training_progress - 仅当prompt明确允许时才使用

禁止使用：
- original_reward - 官方奖励已被屏蔽，不可使用
- official_reward - 任何形式的官方奖励
- 未声明的info字段 - info始终为空
- 未声明的obs切片 - 仅使用obs[0]到obs[7]

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) - 相对于目标的位置
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) - 线速度
- orientation: obs[4] (body_angle) - 姿态角
- angular_velocity: obs[5] - 角速度
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) - 接触标志
- action/engine: action (0-3) - 引擎使用情况，可用于惩罚燃料消耗

## 8. 不确定或不可用的信号
- 终止原因区分：无法区分成功(success)和失败(failure)终止，因为info为空且terminated标志为True时无法知道具体原因
- 目标平台精确位置：obs[0]和obs[1]是相对坐标，但不知道目标平台的具体尺寸和位置阈值
- 稳定状态定义：body_not_awake_or_settled的具体判定标准未知
- 坠毁判定标准：crash_or_body_contact的具体触发条件未知
- 视口边界：horizontal_position_outside_viewport的具体边界值未知
- 初始随机力：初始条件中的随机力大小和方向未知
- 时间步长：无时间步数信息，无法设置时间惩罚
