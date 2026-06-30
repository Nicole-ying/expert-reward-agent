# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器/着陆器轨迹优化任务。智能体控制一个飞行器，从视口顶部中央附近开始，受到初始随机力。目标是尽快到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求"到达并稳定在中央目标平台上"，核心目标是导航到目标位置并稳定着陆，同时优化燃料消耗。这符合导航目标到达任务的定义。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推测）
- obs[0]: x_position — 水平坐标（相对于目标平台中心）
- obs[1]: y_position — 垂直坐标（相对于平台高度）
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体角度（姿态角）
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左侧支撑接触标志（1.0=接触，0.0=未接触）
- obs[7]: right_support_contact — 右侧支撑接触标志（1.0=接触，0.0=未接触）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine — 不执行任何操作
- action 1: left_orientation_engine — 点火左侧姿态发动机（产生顺时针/逆时针力矩）
- action 2: main_engine — 点火主发动机（产生推力）
- action 3: right_orientation_engine — 点火右侧姿态发动机（产生相反方向的力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled — 当飞行器停止运动并稳定在平台上时触发，可能是成功着陆的标志
- failure-like termination: crash_or_body_contact — 发生碰撞或非预期机体接触，可能是坠毁或硬着陆
- failure-like termination: horizontal_position_outside_viewport — 水平位置超出视口范围，可能是飞出边界
- ambiguous termination: 无明确标记为truncation的终止条件

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false — step返回的info为空字典{}，没有显式成功标志
- explicit_failure_flag_available: false — step返回的info为空字典{}，没有显式失败标志
- allowed_info_fields: 无（info为空字典）
- forbidden_or_uncertain_info_fields: 所有info字段（因为info为空）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs — 当前观测（8维向量）
- action — 当前动作（0-3的整数）
- next_obs — 下一时刻观测（8维向量）
- info — 当前为空字典，仅当明确允许时才使用其字段
- training_progress — 仅当prompt明确允许时才使用

禁止使用：
- original_reward — 官方奖励已被屏蔽，禁止使用
- official_reward — 禁止使用
- 未声明的info字段 — info为空，无可用字段
- 未声明的obs切片 — 仅使用上述8个已定义字段

## 7. 可用于奖励函数的信号
- position: obs[0]（x_position）、obs[1]（y_position）— 相对于目标的位置
- velocity: obs[2]（x_velocity）、obs[3]（y_velocity）— 线速度
- orientation: obs[4]（body_angle）— 姿态角
- angular_velocity: obs[5] — 角速度
- contact: obs[6]（left_support_contact）、obs[7]（right_support_contact）— 接触标志
- action/engine: action（0-3）— 引擎使用情况，可用于惩罚燃料消耗

## 8. 不确定或不可用的信号
- 终止条件的具体含义：无法确定body_not_awake_or_settled是成功着陆还是其他情况，因为没有显式success标志
- crash_or_body_contact的具体阈值：不知道什么程度的碰撞算作crash
- 目标平台的精确尺寸和位置：观测是相对坐标，但不知道平台的具体大小
- 初始随机力的分布：不知道初始扰动的具体参数
- 时间步长限制：没有显式的max_steps或truncation信息
- 燃料限制：虽然没有显式燃料状态，但动作空间暗示燃料消耗是优化目标之一
- 奖励函数的任何官方实现细节：已被屏蔽，禁止猜测或重建