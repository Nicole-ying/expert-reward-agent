# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个物体从视口顶部中央附近开始，带有初始随机力。目标是尽快到达并稳定在中央目标平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、降低速度、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求"到达并稳定在中央目标平台上"，且观察空间包含位置、速度、姿态等导航相关特征。动作空间包含方向控制和主引擎，符合导航到达类任务特征。同时包含"尽可能少使用引擎推力"的优化目标，但核心仍是到达目标位置。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推测)
- obs[0]: x_position - 相对于目标平台的水平坐标
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
- action 1: left_orientation_engine - 启动左侧方向引擎
- action 2: main_engine - 启动主引擎
- action 3: right_orientation_engine - 启动右侧方向引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 机体停止运动或稳定在目标上，可能是成功终止
- failure-like termination: crash_or_body_contact - 坠毁或非预期机体接触；horizontal_position_outside_viewport - 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 无 (truncated 始终为 False)

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false (info 字典为空，无显式成功标志)
- explicit_failure_flag_available: false (info 字典为空，无显式失败标志)
- allowed_info_fields: 无 (info 字典为空)
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs (当前状态，8维数组)
- action (当前动作，0-3整数)
- next_obs (下一状态，8维数组)
- info (当前为空字典，仅当明确允许的字段出现时才可使用)
- training_progress (仅当 prompt 明确允许时才用)

禁止使用：
- original_reward (官方奖励已屏蔽)
- official_reward (任何形式的官方奖励重构)
- 未声明的 info 字段
- 未声明的 obs 切片

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) - 相对于目标的位置
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) - 线速度
- orientation: obs[4] (body_angle), obs[5] (angular_velocity) - 姿态角和角速度
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) - 支撑接触标志
- action/engine: action (0-3) - 引擎使用情况，可用于惩罚燃料消耗

## 8. 不确定或不可用的信号
- 终止条件的具体判断阈值：crash_or_body_contact、horizontal_position_outside_viewport、body_not_awake_or_settled 的具体触发条件未知
- 目标平台的精确位置和大小：虽然位置是相对坐标，但目标区域的具体尺寸未知
- 初始随机力的分布和大小：初始条件的具体参数未知
- 引擎推力的具体物理参数：每个动作产生的推力大小和方向未知
- 时间步长和物理模拟参数：dt、重力等物理参数未知
- 任何 info 字段：当前 info 字典为空，无法使用任何额外信号