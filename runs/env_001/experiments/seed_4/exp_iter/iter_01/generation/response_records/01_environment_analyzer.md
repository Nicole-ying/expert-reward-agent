# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器轨迹优化任务。一个飞行器从视口顶部中央附近开始，受到初始随机力的作用。目标是尽可能快地到达并稳定在中央目标平台上，同时使用尽可能少的引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求"到达并稳定在中央目标平台上"，且涉及位置、速度、姿态控制，属于典型的导航到达任务。虽然也涉及推力优化（多目标），但核心目标是到达目标位置并稳定。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推测）
- obs[0]: x_position - 相对于目标平台的水平坐标
- obs[1]: y_position - 相对于平台高度的垂直坐标
- obs[2]: x_velocity - 水平线速度
- obs[3]: y_velocity - 垂直线速度
- obs[4]: body_angle - 机体姿态角
- obs[5]: angular_velocity - 角速度
- obs[6]: left_support_contact - 左侧支撑接触标志（1.0表示接触，0.0表示未接触）
- obs[7]: right_support_contact - 右侧支撑接触标志（1.0表示接触，0.0表示未接触）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 点火左侧姿态引擎
- action 2: main_engine - 点火主引擎
- action 3: right_orientation_engine - 点火右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机体稳定/静止，可能表示成功到达目标并稳定）
- failure-like termination: crash_or_body_contact（坠毁或机体接触）、horizontal_position_outside_viewport（水平位置超出视口）
- ambiguous termination: 无
- truncation: 无（step返回的truncated始终为False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（没有明确的success标志）
- explicit_failure_flag_available: false（没有明确的failure标志）
- allowed_info_fields: 无（info字典为空{}）
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
- info 中明确允许的字段：无
- training_progress 只有 prompt 明确允许时才用

禁止使用：
- original_reward（官方奖励已被屏蔽）
- official_reward（任何形式的官方奖励）
- 未声明的 info 字段（info为空）
- 未声明的 obs 切片

## 7. 可用于奖励函数的信号
- position: obs[0]（x_position）、obs[1]（y_position）
- velocity: obs[2]（x_velocity）、obs[3]（y_velocity）
- orientation: obs[4]（body_angle）
- angular_velocity: obs[5]（angular_velocity）
- contact: obs[6]（left_support_contact）、obs[7]（right_support_contact）
- action/engine: action（0-3，表示引擎选择）

## 8. 不确定或不可用的信号
- 终止原因：无法区分终止是成功（settled）还是失败（crash/out_of_viewport），因为terminated是三个条件的或运算
- 引擎推力大小：动作是离散的，无法知道引擎推力的具体数值
- 目标平台的具体位置：obs[0]和obs[1]已经是相对坐标，无法知道绝对位置
- 时间步数：没有提供当前步数信息
- 燃料/能量消耗：没有提供燃料或能量相关的观测
- 距离目标的精确距离：需要从obs[0]和obs[1]计算
- 接触的物理性质：只知道是否接触，不知道接触力大小
