# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个物体从视口顶部中央附近开始，带有初始随机力。目标是尽快到达并稳定在中央目标着陆台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求"到达并稳定在中央目标着陆台"，核心目标是导航到目标位置并稳定着陆，同时优化燃料消耗。这符合导航目标到达任务的定义，包含位置到达、速度控制和姿态稳定等子目标。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推测)
- obs[0]: x_position - 相对于目标着陆台的水平坐标
- obs[1]: y_position - 相对于着陆台高度的垂直坐标
- obs[2]: x_velocity - 水平线速度
- obs[3]: y_velocity - 垂直线速度
- obs[4]: body_angle - 机体姿态角
- obs[5]: angular_velocity - 角速度
- obs[6]: left_support_contact - 左侧支撑接触标志 (0.0 或 1.0)
- obs[7]: right_support_contact - 右侧支撑接触标志 (0.0 或 1.0)

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 点火左侧姿态引擎
- action 2: main_engine - 点火主引擎
- action 3: right_orientation_engine - 点火右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 当物体停止运动并稳定时，可能表示成功着陆
- failure-like termination: crash_or_body_contact - 坠毁或非预期机体接触；horizontal_position_outside_viewport - 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 无 (truncated 始终为 False)

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false (info 字典为空，无明确成功标志)
- explicit_failure_flag_available: false (info 字典为空，无明确失败标志)
- allowed_info_fields: 无 (info 字典为空)
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs - 当前观测
- action - 当前动作
- next_obs - 下一步观测
- info 中明确允许的字段：无
- training_progress 只有 prompt 明确允许时才用：不适用

禁止使用：
- original_reward - 官方奖励已被屏蔽
- official_reward - 任何形式的官方奖励
- 未声明的 info 字段 - info 字典为空
- 未声明的 obs 切片 - 仅允许使用 obs[0]~obs[7]

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) - 相对于目标的位置
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) - 线速度
- orientation: obs[4] (body_angle) - 姿态角
- angular_velocity: obs[5] (angular_velocity) - 角速度
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) - 接触标志
- action/engine: action (0-3) - 引擎使用情况

## 8. 不确定或不可用的信号
- 原始官方奖励 (original_reward) - 已被屏蔽，禁止使用
- info 字典中的任何字段 - 为空字典，无可用信息
- 训练进度 (training_progress) - 未在 prompt 中明确允许使用
- 任何超出 obs[0]~obs[7] 的观测信息 - 不可用
- 环境内部状态（如风力、引擎推力等） - 不可用
- 成功/失败标志 - 无显式标志，需从观测推断