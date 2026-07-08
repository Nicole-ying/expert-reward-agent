# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
这是一个二维飞行器/着陆器任务。智能体从视口顶部中央附近出发，带有随机初始力。核心目标是**尽快抵达中心目标垫并稳定停靠**，同时尽量**减少引擎推力消耗**。智能体需要学习接近目标、降低速度、保持稳定姿态，并与目标垫安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 浮点数（除接触标志可能是 float 外，但含义为二值）
- obs[0]: `x_position` — 相对于目标垫的水平坐标
- obs[1]: `y_position` — 相对于目标垫高度的高度差
- obs[2]: `x_velocity` — 水平线速度
- obs[3]: `y_velocity` — 垂直线速度
- obs[4]: `body_angle` — 机体朝向角度
- obs[5]: `angular_velocity` — 角速度
- obs[6]: `left_support_contact` — 左侧支撑/接触标志（1.0 表示接触，0.0 表示未接触）
- obs[7]: `right_support_contact` — 右侧支撑/接触标志（1.0 表示接触，0.0 表示未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: `no_engine` — 不开启任何引擎，惯性滑行
- action 1: `left_orientation_engine` — 点燃左侧姿态调整引擎（产生旋转力矩）
- action 2: `main_engine` — 点燃主引擎（产生向上的推力，可能同时影响角速度）
- action 3: `right_orientation_engine` — 点燃右侧姿态调整引擎（产生相反旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` 可能是主体静止/稳定，暗示成功着陆（但需要配合位置判断）；当主体不再移动且位置接近目标垫时，可视作成功
- failure-like termination: `crash_or_body_contact`（非目标垫的碰撞/坠毁）、`horizontal_position_outside_viewport`（水平飞出允许区域）
- ambiguous termination: 仅凭 `body_not_awake_or_settled` 无法直接判定成功或失败，需结合位置、速度、接触等判断；可能包含卡在障碍物上等非成功情况
- truncation: 代码返回 `terminated` 仅由上述条件控制，未提及显式步数截断；但仍可能存在最大步数限制（由环境外部封装，当前片段未体现）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （返回的空字典，无任何可用字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；尤其注意 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 均不存在，严禁假设其存在

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前步的观察数组（含全部8个元素）
- action: 当前步执行的动作编号（int）
- next_obs: 下一步的观察数组（含全部8个元素）
- info: 仅允许使用明确声明的字段（当前为空，故 info 不得用于逻辑判断）
- training_progress: 除非 prompt 明确允许，否则**禁止使用**（本环境卡片未明确允许，因此不可用）

禁止使用：
- original_reward（官方奖励或任何隐藏奖励信号）
- official_reward
- 未声明的 info 字段（包括但不限于 success, failure, termination_reason, distance_remaining 等）
- 未声明的 obs 切片（只能使用索引0-7的完整观察，禁止自行假定其他维度）

## 7. 可用于奖励函数的信号
从 obs 和 next_obs 中可直接提取：
- position: `x_position` (obs[0]), `y_position` (obs[1]) — 可用于计算到目标的距离
- velocity: `x_velocity` (obs[2]), `y_velocity` (obs[3]) — 可用于惩罚过快的速度或鼓励减速
- orientation: `body_angle` (obs[4]), `angular_velocity` (obs[5]) — 可用于鼓励保持稳定姿态
- contact: `left_support_contact` (obs[6]), `right_support_contact` (obs[7]) — 可用于判断是否接触目标垫（两侧同时接触可能表示平稳着陆）
- action/engine: 动作编号可用于惩罚引擎使用（动作0无惩罚，动作1/2/3有惩罚）

## 8. 不确定或不可用的信号
- 官方奖励 `original_reward`（已屏蔽）
- 明确的 success / failure 布尔标志（info 中不存在）
- 目标垫的精确坐标（仅能从相对位置推断）
- 真实环境名称、基准名称
- 剩余步数或最大步数（未在片段中暴露）
- 外部风力、引擎冲量等物理参数（被屏蔽）
- 任何通过 info 传递的辅助度量（如 fuel_consumption, distance 等）
