# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器轨迹优化任务。一个飞行器从视口顶部中央附近开始，受到初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求"到达并稳定在中央目标着陆平台"，涉及位置到达、速度控制、姿态稳定，属于典型的导航到达任务。同时包含燃料效率优化（使用尽可能少的引擎推力），但核心目标是到达目标位置。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推测）
- obs[0]: x_position - 相对于目标着陆平台的水平坐标
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
- success-like termination: body_not_awake_or_settled - 机体停止运动或稳定着陆，可能是成功到达目标
- failure-like termination: crash_or_body_contact - 坠毁或机体接触（非正常着陆），horizontal_position_outside_viewport - 水平位置超出视口范围
- ambiguous termination: 无
- truncation: 无（step返回的truncated始终为False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false - 没有明确的成功标志
- explicit_failure_flag_available: false - 没有明确的失败标志
- allowed_info_fields: 无（info始终为空字典{}）
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
- info - 空字典，不可用
- training_progress - 仅当prompt明确允许时才使用

禁止使用：
- original_reward - 官方奖励已被屏蔽
- official_reward - 任何形式的官方奖励
- 未声明的info字段 - info始终为空
- 未声明的obs切片 - 仅使用已定义的8个维度

## 7. 可用于奖励函数的信号
- position: obs[0] x_position, obs[1] y_position - 相对于目标的位置
- velocity: obs[2] x_velocity, obs[3] y_velocity - 线速度
- orientation: obs[4] body_angle, obs[5] angular_velocity - 姿态角和角速度
- contact: obs[6] left_support_contact, obs[7] right_support_contact - 支撑接触标志
- action/engine: action - 引擎使用情况（可用于惩罚燃料消耗）

## 8. 不确定或不可用的信号
- 目标平台的具体位置和大小：观测是相对坐标，但绝对尺寸未知
- 坠毁/接触的具体判定阈值：crash_or_body_contact的具体条件未知
- 稳定着陆的具体判定标准：body_not_awake_or_settled的具体条件未知
- 视口边界的具体范围：horizontal_position_outside_viewport的边界值未知
- 初始随机力的具体参数：初始扰动的大小和方向未知
- 引擎推力的具体数值：各引擎产生的力和力矩大小未知
- 时间步长或物理模拟参数：dt等参数未知
- 任何info字段：info始终为空字典，不可用
