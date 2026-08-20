# Env_001 环境理解卡片

## 1. 任务目标
这是一个二维飞行器轨迹优化任务。飞行器从视口顶部附近开始，受到随机初始力。它必须尽快到达并稳定地降落在中央目标平台上，同时尽可能少地使用发动机推力。智能体需要学会靠近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（假设，环境未指定但通常如此）
- obs[0]: x_position，飞行器相对于目标垫的水平坐标
- obs[1]: y_position，飞行器相对于平台高度的垂直坐标
- obs[2]: x_velocity，水平线速度
- obs[3]: y_velocity，垂直线速度
- obs[4]: body_angle，机体方向角
- obs[5]: angular_velocity，角速度
- obs[6]: left_support_contact，左侧支撑腿接触标志（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact，右侧支撑腿接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，不启动任何引擎（滑行）
- action 1: left_orientation_engine，启动左侧姿态引擎
- action 2: main_engine，启动主引擎（向下喷气产生向上推力）
- action 3: right_orientation_engine，启动右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 且飞行器位置在目标平台上方，左右支撑腿均接触，速度接近零。这表示安全着陆并稳定。
- failure-like termination: crash_or_body_contact（机体非腿部部分与地面或其他障碍物碰撞），或者 horizontal_position_outside_viewport（水平飞出视口边界）。
- ambiguous termination: body_not_awake_or_settled 也可能由坠毁后静止触发，此时需结合位置和接触腿信息判断。
- truncation: 环境未提及时间截断，但可能由封装器实现（本描述不涉及）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 字典为空）
- forbidden_or_uncertain_info_fields: 任何 info 字段（包括 potential success、failure、termination_reason 等）均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前观察（8维数组）
- action：当前执行的动作（int）
- next_obs：下一时刻观察（8维数组）
- info 中明确允许的字段：无（info 始终为空字典）
- training_progress：只有在 prompt 明确允许时才使用（此处未明确允许，避免使用）

禁止使用：
- original_reward（官方奖励被屏蔽，不得参考或使用）
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片

## 7. 可用于奖励函数的信号
- position: obs[0]（水平坐标）、obs[1]（垂直坐标）
- velocity: obs[2]（水平速度）、obs[3]（垂直速度）
- orientation: obs[4]（机体角度）、obs[5]（角速度）
- contact: obs[6]（左腿接触）、obs[7]（右腿接触）
- action/engine: action 的取值（0~3），可推断推力使用

## 8. 不确定或不可用的信号
- 明确的成功/失败标志：不存在
- 目标是否已到达：只能通过位置、速度、接触等信息间接推断
- 剩余燃料量：未提供
- 时间/步数计数：未作为观察给出
- 外部风速或随机力：不可观测