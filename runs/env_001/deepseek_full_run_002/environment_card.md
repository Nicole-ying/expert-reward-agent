# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个物体从视口顶部中央附近开始，带有初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求"到达并稳定在中央目标着陆平台"，核心是导航到目标位置并稳定着陆，属于典型的导航目标到达任务。同时包含"尽可能少使用引擎推力"的优化目标，但主要任务类型仍是导航到达。

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
- obs[6]: left_support_contact - 左侧支撑接触标志（1.0表示接触，0.0表示未接触）
- obs[7]: right_support_contact - 右侧支撑接触标志（1.0表示接触，0.0表示未接触）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine - 不做任何操作
- action 1: left_orientation_engine - 点火左侧姿态发动机
- action 2: main_engine - 点火主发动机
- action 3: right_orientation_engine - 点火右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 机体不再运动或已稳定，可能表示成功着陆
- failure-like termination: crash_or_body_contact - 坠毁或机体接触（非正常着陆接触），horizontal_position_outside_viewport - 水平位置超出视口范围
- ambiguous termination: 无
- truncation: False（step返回的truncated固定为False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info字典为空{}，无显式成功标志）
- explicit_failure_flag_available: false（info字典为空{}，无显式失败标志）
- allowed_info_fields: 无（info字典为空）
- forbidden_or_uncertain_info_fields: 所有info字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs - 当前观测（8维向量）
- action - 当前动作（0-3整数）
- next_obs - 下一时刻观测（8维向量）
- info - 当前为空字典{}，无可用字段
- training_progress - 仅当prompt明确允许时才使用

禁止使用：
- original_reward - 官方奖励已被屏蔽，禁止使用
- official_reward - 禁止使用
- 未声明的info字段 - info为空，无可用字段
- 未声明的obs切片 - 仅允许使用上述8个已定义字段

## 7. 可用于奖励函数的信号
- position: obs[0]（x_position），obs[1]（y_position）- 相对于目标的位置
- velocity: obs[2]（x_velocity），obs[3]（y_velocity）- 线速度
- orientation: obs[4]（body_angle），obs[5]（angular_velocity）- 姿态角和角速度
- contact: obs[6]（left_support_contact），obs[7]（right_support_contact）- 支撑接触标志
- action/engine: action（0-3整数）- 当前选择的发动机动作

## 8. 不确定或不可用的信号
- 终止条件的具体含义：crash_or_body_contact 无法区分是"坠毁"还是"正常着陆接触"，body_not_awake_or_settled 无法区分是"成功稳定"还是"意外停止"
- 目标着陆平台的具体位置：obs[0]和obs[1]是相对坐标，但不知道目标范围/阈值
- 初始随机力的方向和大小：未在观测中提供
- 引擎推力的大小和持续时间：动作是离散的，但推力大小未知
- 时间步数或最大步长限制：未提供
- 燃料/能量消耗信息：未在观测中提供
- 风速或其他环境干扰：未在观测中提供
- 着陆成功的具体判定标准：未提供阈值或判定条件