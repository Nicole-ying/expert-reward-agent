# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器轨迹优化任务。一个飞行器从视口顶部中央附近开始，受到初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求"到达并稳定在中央目标着陆平台"，属于典型的导航到达任务。同时包含速度控制、姿态稳定等子目标，但核心是到达目标位置。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推测)
- obs[0]: x_position - 相对于目标着陆平台的水平坐标
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
- action 1: left_orientation_engine - 点火左侧姿态发动机
- action 2: main_engine - 点火主发动机
- action 3: right_orientation_engine - 点火右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 机体停止运动或稳定在目标上，可能是成功到达
- failure-like termination: crash_or_body_contact - 坠毁或非预期机体接触；horizontal_position_outside_viewport - 水平位置超出视口
- ambiguous termination: 无
- truncation: 无 (truncated 始终为 False)

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false (info 字典为空，无明确成功标志)
- explicit_failure_flag_available: false (info 字典为空，无明确失败标志)
- allowed_info_fields: 无 (info 始终为 {})
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs (当前观测)
- action (当前动作)
- next_obs (下一观测)
- training_progress (仅当 prompt 明确允许时才使用)

禁止使用：
- original_reward (官方奖励已屏蔽)
- info (始终为空字典)
- 未声明的 obs 切片

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) - 相对于目标的位置
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) - 线速度
- orientation: obs[4] (body_angle), obs[5] (angular_velocity) - 姿态和角速度
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) - 支撑接触标志
- action/engine: action (0-3) - 引擎使用情况

## 8. 不确定或不可用的信号
- 绝对位置/坐标：观测中只有相对位置，无绝对坐标
- 目标平台尺寸/形状：未在观测中提供
- 引擎推力大小：动作是离散的，无连续推力值
- 燃料/能量消耗：未在观测中提供
- 时间步/步数：未在观测或 info 中提供
- 成功/失败标志：info 为空，无明确成功或失败信号
- 坠毁类型：crash_or_body_contact 的具体含义不明确
- 初始随机力：不可观测
- 视口边界：未在观测中提供