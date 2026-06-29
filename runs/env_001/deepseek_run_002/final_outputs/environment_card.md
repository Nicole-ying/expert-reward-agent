# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。一个飞行器从视口顶部附近开始，带有初始随机力。目标是尽可能快地到达并稳定在中央目标着陆台上，同时使用尽可能少的引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求"到达并稳定在中央目标着陆台"，核心是导航到目标位置并稳定停靠，同时优化燃料消耗。这符合导航目标到达任务的核心特征——到达特定目标位置并保持稳定状态。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推测)
- obs[0]: x_position — 相对于目标着陆台的水平坐标
- obs[1]: y_position — 相对于着陆台高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 飞行器姿态角
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左侧支撑接触标志 (0.0 或 1.0)
- obs[7]: right_support_contact — 右侧支撑接触标志 (0.0 或 1.0)

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不执行任何操作
- action 1: left_orientation_engine — 启动左侧姿态引擎
- action 2: main_engine — 启动主引擎
- action 3: right_orientation_engine — 启动右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled — 当飞行器在着陆台上稳定停靠时触发，可能是成功终止
- failure-like termination: crash_or_body_contact — 坠毁或非预期身体接触，明显是失败终止
- failure-like termination: horizontal_position_outside_viewport — 水平位置超出视口边界，明显是失败终止
- ambiguous termination: 无
- truncation: 无显式截断，但可能由环境内部处理

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false — step 返回的 info 为空字典，无显式成功标志
- explicit_failure_flag_available: false — step 返回的 info 为空字典，无显式失败标志
- allowed_info_fields: 无（info 为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段（因为 info 为空字典）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs — 当前状态观测
- action — 当前执行的动作
- next_obs — 执行动作后的下一状态观测
- info — 当前为空字典，无可用字段
- training_progress — 只有 prompt 明确允许时才用

禁止使用：
- original_reward — 官方奖励已被屏蔽，禁止使用
- official_reward — 禁止使用
- 未声明的 info 字段 — info 为空字典
- 未声明的 obs 切片 — 仅允许使用 obs[0]~obs[7]

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) — 相对于目标的位置
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) — 线速度
- orientation: obs[4] (body_angle), obs[5] (angular_velocity) — 姿态角和角速度
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) — 接触标志
- action/engine: action (0~3) — 动作选择，可用于惩罚引擎使用

## 8. 不确定或不可用的信号
- 目标着陆台的具体位置和尺寸 — 未在观测中提供
- 飞行器的质量、惯性等物理参数 — 未在观测中提供
- 引擎推力大小和方向 — 未在观测中提供
- 时间步长或剩余时间 — 未在观测中提供
- 燃料剩余量 — 未在观测中提供
- 风速或环境扰动 — 未在观测中提供
- 着陆台高度或地面高度 — 未在观测中提供
- 任何 info 字段 — info 为空字典