# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个物体从视口顶部中央附近开始，带有初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、降低速度、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求"到达并稳定在中央目标着陆平台"，核心目标是导航到目标位置并稳定停靠，同时优化燃料消耗（引擎推力使用）。这符合导航到达任务的核心特征：目标位置到达 + 速度/姿态控制。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推测）
- obs[0]: x_position - 相对于目标着陆平台的水平坐标
- obs[1]: y_position - 相对于着陆平台高度的垂直坐标
- obs[2]: x_velocity - 水平线速度
- obs[3]: y_velocity - 垂直线速度
- obs[4]: body_angle - 机体姿态角
- obs[5]: angular_velocity - 角速度
- obs[6]: left_support_contact - 左侧支撑接触标志（1.0=接触，0.0=未接触）
- obs[7]: right_support_contact - 右侧支撑接触标志（1.0=接触，0.0=未接触）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 点火左侧姿态发动机
- action 2: main_engine - 点火主发动机
- action 3: right_orientation_engine - 点火右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 当机体停止运动并稳定在目标位置时触发，可能是成功终止
- failure-like termination: crash_or_body_contact - 坠毁或机体接触地面（非目标平台），可能是失败终止
- ambiguous termination: horizontal_position_outside_viewport - 水平位置超出视口边界，可能是失败或截断
- truncation: 无显式截断条件，但视口边界外可视为截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false - step返回的info为空字典{}，无显式成功标志
- explicit_failure_flag_available: false - step返回的info为空字典{}，无显式失败标志
- allowed_info_fields: 无 - info字典为空，无可用字段
- forbidden_or_uncertain_info_fields: 所有info字段均不可用，因为step返回空字典

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs - 当前步的观测（8维向量）
- action - 当前步执行的动作（0-3整数）
- next_obs - 下一步的观测（8维向量）
- info - 当前为空字典，无可用字段
- training_progress - 仅当prompt明确允许时才使用

禁止使用：
- original_reward - 官方奖励已被屏蔽，禁止使用
- official_reward - 禁止重建官方奖励
- 未声明的info字段 - info为空字典，无可用字段
- 未声明的obs切片 - 仅允许使用obs[0]到obs[7]的明确定义字段

## 7. 可用于奖励函数的信号
- position: obs[0]（x_position），obs[1]（y_position）- 相对于目标的位置
- velocity: obs[2]（x_velocity），obs[3]（y_velocity）- 线速度
- orientation: obs[4]（body_angle），obs[5]（angular_velocity）- 姿态和角速度
- contact: obs[6]（left_support_contact），obs[7]（right_support_contact）- 支撑接触标志
- action/engine: action（0-3）- 引擎使用情况，可用于惩罚燃料消耗

## 8. 不确定或不可用的信号
- 终止原因：无法区分成功终止（settled）和失败终止（crash/out_of_bounds），因为info为空字典
- 目标平台接触标志：只有左右支撑接触标志，没有明确的目标平台接触检测
- 燃料/引擎使用次数：动作空间中有引擎动作，但观测中没有燃料剩余量信息
- 时间步/步数限制：无显式步数限制信息
- 距离目标的绝对距离：只有相对位置，没有直接的距离度量
- 成功/失败标志：step返回空info，无法直接获取成功或失败信号