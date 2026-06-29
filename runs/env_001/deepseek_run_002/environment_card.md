# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个飞行器从视口顶部附近开始，带有初始随机力。目标是尽可能快地到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、降低速度、保持稳定姿态，并实现安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求"到达并稳定在中央目标着陆平台"，核心目标是导航到目标位置并稳定着陆，属于典型的导航目标到达任务。同时包含速度控制、姿态稳定等子目标，但主要目标是到达目标位置。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推测)
- obs[0]: x_position - 相对于目标着陆平台的水平坐标
- obs[1]: y_position - 相对于着陆平台高度的垂直坐标
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
- success-like termination: body_not_awake_or_settled - 当飞行器停止运动并稳定在着陆平台上时触发，可能是成功着陆的标志
- failure-like termination: crash_or_body_contact - 发生碰撞或非正常机体接触，可能是坠毁或硬着陆
- failure-like termination: horizontal_position_outside_viewport - 水平位置超出视口范围，可能是飞离目标区域
- ambiguous termination: 无明确区分成功/失败的终止标志

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false - step返回的info为空字典，没有显式成功标志
- explicit_failure_flag_available: false - step返回的info为空字典，没有显式失败标志
- allowed_info_fields: 无（info为空字典）
- forbidden_or_uncertain_info_fields: 所有info字段（因为info为空字典）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs - 当前状态观测
- action - 当前执行的动作
- next_obs - 执行动作后的下一状态观测
- info - 当前为空字典，但可保留接口

禁止使用：
- original_reward - 官方奖励已被屏蔽，禁止使用
- training_progress - 除非prompt明确允许，否则禁止使用
- 未声明的info字段 - info为空字典，无可用字段

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) - 可用于计算到目标的距离
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) - 可用于鼓励减速或稳定
- orientation: obs[4] (body_angle), obs[5] (angular_velocity) - 可用于鼓励稳定姿态
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) - 可用于检测是否成功着陆
- action: action (0-3) - 可用于惩罚引擎使用，鼓励节能

## 8. 不确定或不可用的信号
- 原始奖励值 (original_reward) - 已被屏蔽，禁止使用
- 任何info字段 - info为空字典，无可用信息
- 训练进度 (training_progress) - 除非明确允许，否则不可用
- 环境内部状态 - 如引擎推力、风力等内部物理参数不可用
- 时间步计数 - 未在观测空间中提供
- 目标位置绝对坐标 - 观测是相对坐标，绝对位置不可用