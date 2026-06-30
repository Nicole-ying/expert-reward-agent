# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个飞行器从视口顶部中央附近开始，受到初始随机力的作用。目标是尽可能快地到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、降低速度、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求"到达并稳定在中央目标着陆平台"，这是一个典型的导航到达目标点任务。同时包含速度控制、姿态稳定和燃料效率优化，但核心目标是到达目标位置并稳定着陆。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推测)
- obs[0]: x_position - 相对于目标着陆垫的水平坐标
- obs[1]: y_position - 相对于着陆垫高度的垂直坐标
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
- success-like termination: body_not_awake_or_settled - 机体不再运动或已稳定，可能表示成功着陆
- failure-like termination: crash_or_body_contact - 坠毁或机体接触（非正常着陆接触）
- failure-like termination: horizontal_position_outside_viewport - 水平位置超出视口范围（飞出边界）
- ambiguous termination: 无明确区分成功/失败的终止标志
- truncation: False (step返回的truncated始终为False)

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false - 没有明确的成功标志
- explicit_failure_flag_available: false - 没有明确的失败标志
- allowed_info_fields: {} - info字典为空，无可用字段
- forbidden_or_uncertain_info_fields: 所有info字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs - 当前状态观测
- action - 当前执行的动作
- next_obs - 执行动作后的下一状态观测
- info - 当前为空字典，无可用字段
- training_progress - 仅当prompt明确允许时才使用

禁止使用：
- original_reward - 官方奖励已被屏蔽，禁止使用
- official_reward - 禁止使用
- 未声明的info字段 - info为空字典
- 未声明的obs切片 - 仅允许使用obs[0]到obs[7]的8个维度

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) - 相对于目标的位置
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) - 线速度
- orientation: obs[4] (body_angle) - 姿态角
- angular_velocity: obs[5] (angular_velocity) - 角速度
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) - 支撑接触标志
- action: action (0-3) - 动作选择，可用于惩罚燃料消耗

## 8. 不确定或不可用的信号
- 终止条件的具体含义：无法确定body_not_awake_or_settled是成功着陆还是其他情况
- 目标着陆垫的具体位置：obs[0]和obs[1]是相对坐标，但不知道目标范围大小
- 接触标志的物理含义：左右支撑接触标志的具体物理意义不明确（是着陆腿接触还是其他）
- 初始随机力的具体参数：未知
- 视口边界的具体范围：未知
- 成功着陆的判定标准：未知（位置精度、速度阈值、姿态角范围等）
- 燃料消耗的具体量化：动作选择与燃料消耗的关系未知
- 时间步长限制：未提供最大步数信息
