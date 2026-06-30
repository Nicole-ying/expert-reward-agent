# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器轨迹优化任务。一个飞行器从视口顶部中央附近开始，受到初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求"到达并稳定在中央目标着陆平台"，涉及位置到达、速度控制、姿态稳定，属于典型的导航到达任务。同时包含燃料效率优化（使用尽可能少的引擎推力），但核心目标是到达目标位置。

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
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 点火左侧姿态发动机
- action 2: main_engine - 点火主发动机
- action 3: right_orientation_engine - 点火右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机体稳定/静止在目标区域，可能表示成功着陆）
- failure-like termination: crash_or_body_contact（坠毁或机体接触地面，可能表示失败）
- ambiguous termination: horizontal_position_outside_viewport（水平位置超出视口，可能是失败或边界条件）
- truncation: 无显式截断（step返回的truncated固定为False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info字典为空{}，无显式成功标志）
- explicit_failure_flag_available: false（info字典为空{}，无显式失败标志）
- allowed_info_fields: 无（info为空字典）
- forbidden_or_uncertain_info_fields: 所有info字段（因为info为空）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前状态，8维向量）
- action（当前动作，0-3的整数）
- next_obs（下一状态，8维向量）
- info（当前为空字典，仅当明确允许的字段出现时才可使用）
- training_progress（仅当prompt明确允许时才使用）

禁止使用：
- original_reward（官方奖励已被屏蔽，不可重构）
- official_reward（任何形式的官方奖励）
- 未声明的info字段（info为空）
- 未声明的obs切片（仅可使用上述8个字段）

## 7. 可用于奖励函数的信号
- position: obs[0]（x_position）、obs[1]（y_position）——可用于计算与目标的距离
- velocity: obs[2]（x_velocity）、obs[3]（y_velocity）——可用于鼓励减速
- orientation: obs[4]（body_angle）——可用于鼓励稳定姿态
- angular_velocity: obs[5]（angular_velocity）——可用于鼓励角速度归零
- contact: obs[6]（left_support_contact）、obs[7]（right_support_contact）——可用于检测是否成功着陆
- action: action（0-3）——可用于惩罚引擎使用，鼓励燃料效率

## 8. 不确定或不可用的信号
- 终止条件的具体阈值：crash_or_body_contact、horizontal_position_outside_viewport、body_not_awake_or_settled的具体判断条件未知
- 目标平台的精确位置和尺寸：只知道obs[0]和obs[1]是相对坐标，但具体目标范围未知
- 初始随机力的方向和大小：任务描述提到有初始随机力，但具体参数未知
- 引擎推力的具体物理参数：各引擎的推力大小、燃料消耗等未知
- 成功着陆的精确判定标准：body_not_awake_or_settled的具体含义（速度阈值、位置范围等）未知
- 坠毁的判定标准：crash_or_body_contact的具体条件（接触力阈值、角度范围等）未知
- 视口边界的具体范围：horizontal_position_outside_viewport的边界值未知
- 时间步限制：无显式时间步截断，但可能有隐式限制
- 任何info字段：当前info为空字典，无法获取额外信号
