# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个物体从视口顶部中央附近开始，带有初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求"到达并稳定在中央目标着陆平台"，具有明确的目标位置（着陆平台），同时涉及速度控制、姿态稳定和燃料效率优化，属于典型的导航到达任务。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推断）
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
- success-like termination: body_not_awake_or_settled - 机体停止运动并稳定，可能表示成功着陆
- failure-like termination: crash_or_body_contact - 坠毁或非正常机体接触；horizontal_position_outside_viewport - 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 无（truncated 始终为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空，无显式成功标志）
- explicit_failure_flag_available: false（info 字典为空，无显式失败标志）
- allowed_info_fields: 无（info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前状态）
- action（当前动作）
- next_obs（下一状态）
- info 中明确允许的字段（当前无可用字段）
- training_progress 只有 prompt 明确允许时才用

禁止使用：
- original_reward（官方奖励已屏蔽）
- official_reward（禁止重构）
- 未声明的 info 字段
- 未声明的 obs 切片

## 7. 可用于奖励函数的信号
- position: obs[0] x_position, obs[1] y_position - 可用于计算与目标点的距离
- velocity: obs[2] x_velocity, obs[3] y_velocity - 可用于鼓励减速
- orientation: obs[4] body_angle - 可用于鼓励稳定姿态
- angular_velocity: obs[5] angular_velocity - 可用于鼓励角速度归零
- contact: obs[6] left_support_contact, obs[7] right_support_contact - 可用于检测是否成功着陆
- action: action 值（0-3）- 可用于惩罚引擎使用以鼓励燃料效率

## 8. 不确定或不可用的信号
- 终止条件的具体含义：无法区分 crash_or_body_contact 中的"坠毁"和"正常接触"，body_not_awake_or_settled 的具体物理含义不明确
- 目标平台的精确位置：obs[0] 和 obs[1] 是相对坐标，但不知道目标平台的具体尺寸和允许的着陆误差范围
- 初始随机力的方向和大小：无法从观测中获取
- 引擎推力的具体物理参数：推力大小、燃料消耗率等未知
- 时间步长或物理模拟参数：无法获取
- 任何 info 字段：info 字典为空，无额外信号可用