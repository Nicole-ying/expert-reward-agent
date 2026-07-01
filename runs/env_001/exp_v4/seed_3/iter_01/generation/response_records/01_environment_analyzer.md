# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个物体从视口顶部中央附近开始，带有初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、降低速度、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求"到达并稳定在中央目标着陆平台"，核心是导航到目标位置并稳定停靠，同时优化燃料消耗。这符合导航目标到达任务的定义。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推断)
- obs[0]: x_position — 相对于目标着陆平台的水平坐标
- obs[1]: y_position — 相对于平台高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体朝向角度
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左侧支撑接触标志（0或1）
- obs[7]: right_support_contact — 右侧支撑接触标志（0或1）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine — 不做任何操作
- action 1: left_orientation_engine — 启动左侧姿态引擎
- action 2: main_engine — 启动主引擎
- action 3: right_orientation_engine — 启动右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled — 机体停止运动并稳定在平台上，可能是成功着陆的标志
- failure-like termination: crash_or_body_contact — 坠毁或非正常机体接触；horizontal_position_outside_viewport — 水平位置超出视口范围
- ambiguous termination: 无
- truncation: 无（step返回的truncated始终为False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false — 没有明确的success标志
- explicit_failure_flag_available: false — 没有明确的failure标志
- allowed_info_fields: 无（info始终为空字典{}）
- forbidden_or_uncertain_info_fields: 所有info字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs — 当前观测（8维向量）
- action — 当前动作（0-3整数）
- next_obs — 下一时刻观测（8维向量）
- info — 空字典，不可用
- training_progress — 仅当prompt明确允许时才使用

禁止使用：
- original_reward — 官方奖励已被屏蔽
- official_reward — 任何形式的官方奖励
- 未声明的info字段 — info始终为空
- 未声明的obs切片 — 只能使用已定义的8个维度

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) — 相对于目标的位置
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) — 线速度
- orientation: obs[4] (body_angle) — 机体朝向角度
- angular_velocity: obs[5] — 角速度
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) — 支撑接触标志
- action/engine: action (0-3) — 引擎使用情况，可用于惩罚燃料消耗

## 8. 不确定或不可用的信号
- 目标平台的具体位置和尺寸 — 未在观测中提供
- 着陆成功的精确判定条件 — 只知道"body_not_awake_or_settled"是终止条件之一
- 坠毁的具体判定逻辑 — 只知道"crash_or_body_contact"是终止条件
- 视口边界的具体范围 — 只知道超出会导致终止
- 初始随机力的具体参数 — 未提供
- 引擎推力的具体物理参数 — 未提供
- 时间步长或物理模拟参数 — 未提供
- 任何info字段 — info始终为空字典
