# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个物体从视口顶部中央附近开始，带有初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求"到达并稳定在中央目标着陆平台"，核心目标是导航到目标位置并稳定着陆，同时优化燃料消耗。这符合导航到达任务的定义，其中包含位置到达、速度控制和姿态稳定等子目标。

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
- action 1: left_orientation_engine - 点火左侧姿态引擎
- action 2: main_engine - 点火主引擎
- action 3: right_orientation_engine - 点火右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 当机体停止运动并稳定在目标上时，可能表示成功着陆
- failure-like termination: crash_or_body_contact - 坠毁或非预期的机体接触；horizontal_position_outside_viewport - 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 无（step返回的truncated始终为False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false - 没有明确的成功标志
- explicit_failure_flag_available: false - 没有明确的失败标志
- allowed_info_fields: 无（step返回空字典{}）
- forbidden_or_uncertain_info_fields: 所有info字段均不可用，因为step返回空字典

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs - 当前观测（8维向量）
- action - 当前动作（0-3整数）
- next_obs - 下一时刻观测（8维向量）
- info - 空字典，不可用
- training_progress - 仅当prompt明确允许时才使用

禁止使用：
- original_reward - 官方奖励已被屏蔽
- official_reward - 任何形式的官方奖励
- 未声明的info字段 - info为空字典
- 未声明的obs切片 - 仅使用已定义的8个维度

## 7. 可用于奖励函数的信号
- position: obs[0]（x_position），obs[1]（y_position）- 相对于目标的位置
- velocity: obs[2]（x_velocity），obs[3]（y_velocity）- 线速度
- orientation: obs[4]（body_angle），obs[5]（angular_velocity）- 姿态和角速度
- contact: obs[6]（left_support_contact），obs[7]（right_support_contact）- 支撑接触标志
- action/engine: action（0-3）- 引擎使用情况，可用于惩罚燃料消耗

## 8. 不确定或不可用的信号
- 目标位置的具体坐标：obs[0]和obs[1]是相对值，但不知道目标的具体绝对位置
- 着陆平台尺寸：不知道平台的大小范围
- 成功着陆的判定阈值：不知道位置、速度、姿态的精确成功条件
- 坠毁判定条件：不知道crash_or_body_contact的具体触发条件
- 时间限制：没有明确的步数限制信息
- 初始随机力的大小和方向：无法从观测中获取
- 引擎推力大小：不知道每个引擎动作产生的具体物理效果
- 燃料量：没有燃料限制的观测信息
- 风速等环境干扰：没有相关观测信息