# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器轨迹优化任务。一个飞行器从视口顶部中央附近开始，受到初始随机力。目标是尽可能快地到达并稳定在中央目标着陆台上，同时尽可能少地使用引擎推力。智能体需要学习接近目标、减速、保持稳定姿态并安全着陆。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求"到达并稳定在中央目标着陆台"，涉及位置导航、速度控制、姿态稳定，属于典型的导航到达任务。同时包含燃料效率优化（推力最小化），但核心目标是到达目标位置。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推测）
- obs[0]: x_position - 相对于目标着陆台的水平坐标
- obs[1]: y_position - 相对于着陆台高度的垂直坐标
- obs[2]: x_velocity - 水平线速度
- obs[3]: y_velocity - 垂直线速度
- obs[4]: body_angle - 机体姿态角
- obs[5]: angular_velocity - 角速度
- obs[6]: left_support_contact - 左侧支撑接触标志（0.0或1.0）
- obs[7]: right_support_contact - 右侧支撑接触标志（0.0或1.0）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 点火左侧姿态发动机
- action 2: main_engine - 点火主发动机
- action 3: right_orientation_engine - 点火右侧姿态发动机

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机体稳定/静止，可能表示成功着陆）
- failure-like termination: crash_or_body_contact（坠毁或机体接触）、horizontal_position_outside_viewport（水平位置超出视口）
- ambiguous termination: 无
- truncation: 无（step返回truncated=False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info为空字典{}，无显式成功标志）
- explicit_failure_flag_available: false（info为空字典{}，无显式失败标志）
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
- info 中明确允许的字段（当前无）
- training_progress 只有 prompt 明确允许时才用

禁止使用：
- original_reward（官方奖励已屏蔽）
- official_reward（任何形式的官方奖励重构）
- 未声明的 info 字段（info始终为空）
- 未声明的 obs 切片

## 7. 可用于奖励函数的信号
- position: obs[0]（x_position）、obs[1]（y_position）——可用于计算到目标的距离
- velocity: obs[2]（x_velocity）、obs[3]（y_velocity）——可用于惩罚速度或奖励减速
- orientation: obs[4]（body_angle）——可用于奖励稳定姿态
- angular_velocity: obs[5]——可用于惩罚旋转或奖励稳定
- contact: obs[6]（left_support_contact）、obs[7]（right_support_contact）——可用于检测着陆接触
- action/engine: action（0-3）——可用于惩罚引擎使用（燃料效率）

## 8. 不确定或不可用的信号
- 终止原因：无法区分成功（settled）和失败（crash/out_of_viewport），因为info为空
- 时间步长：无时间步信息，无法直接施加时间惩罚
- 燃料/能量：无显式燃料状态，只能通过动作惩罚间接优化
- 目标位置绝对坐标：obs[0]和obs[1]是相对坐标，但足够用于导航
- 着陆台尺寸/接触阈值：未知，需要从接触标志推断
- 初始随机力：不可观测，但影响初始状态