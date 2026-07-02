# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
本环境是一个 2D 飞行器（类似于月球着陆器）轨迹优化问题。飞行器从视口顶部中央附近以随机初始受力开始运动。  
智能体需要**尽快降落到中央的目标平台上并稳定着陆**，同时**尽量减少发动机推力的使用**。  
期望行为：接近目标、降低速度、保持姿态稳定、安全接触目标平台。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务核心是控制飞行器到达并稳定在目标垫上，属于典型的导航‑目标到达问题（带能耗约束）。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（假定默认）
- obs[0]: `x_position` — 飞行器相对于目标平台的水平坐标
- obs[1]: `y_position` — 飞行器相对于平台高度的垂直坐标
- obs[2]: `x_velocity` — 水平线速度
- obs[3]: `y_velocity` — 垂直线速度
- obs[4]: `body_angle` — 飞行器姿态角
- obs[5]: `angular_velocity` — 角速度
- obs[6]: `left_support_contact` — 左侧支撑接触标志（1.0 接触，0.0 未接触）
- obs[7]: `right_support_contact` — 右侧支撑接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: `no_engine` — 无推力，仅靠惯性运动
- action 1: `left_orientation_engine` — 点燃左姿态调整发动机（改变角速度）
- action 2: `main_engine` — 点燃主发动机（产生竖直向上的推力）
- action 3: `right_orientation_engine` — 点燃右姿态调整发动机（相反方向改变角速度）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` — 飞行器不再活跃或已稳定（可能表示成功着陆并稳定在目标平台）
- failure-like termination:  
  `crash_or_body_contact` — 坠毁或与非目标表面的身体接触  
  `horizontal_position_outside_viewport` — 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 环境源码未出现时间截断，但实际 gym 封装可能有默认时间限制（未提供细节，故在 reward 设计中不应依赖）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 中无 `success` 字段）
- explicit_failure_flag_available: false （info 中无 `failure` 字段）
- allowed_info_fields: 无（info 为空字典 `{}`，不会提供可用的额外信号）
- forbidden_or_uncertain_info_fields: `success`, `failure`, `termination_reason` 等任何未在源码中出现的字段

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs` — 执行动作前的观察（8 维数组）
- `action` — 执行的离散动作编号（0,1,2,3）
- `next_obs` — 动作执行后的观察（8 维数组）
- `info` 中明确允许的字段：无（info 恒为空，不可用）

禁止使用：
- `original_reward` / `official_reward` — 已被掩蔽，不可重建或直接使用
- `training_progress` — 任务描述未明确允许使用，禁止
- 任何未在“允许使用”中列出的 `info` 或 `obs` 派生量

## 7. 可用于奖励函数的信号
- **位置信息**：`next_obs[0]`（相对目标垫水平位置）、`next_obs[1]`（相对目标垫垂直位置）
- **速度信息**：`next_obs[2]`（水平速度）、`next_obs[3]`（垂直速度）
- **姿态信息**：`next_obs[4]`（姿态角）、`next_obs[5]`（角速度）
- **接触信息**：`next_obs[6]`（左支撑接触）、`next_obs[7]`（右支撑接触）
- **动作/引擎使用**：`action` 是否为 1,2,3（引擎激活）可构造能耗惩罚

## 8. 不确定或不可用的信号
- 官方原始奖励：被掩蔽，不可用
- `info` 中的任何字段：不可用（空字典）
- `training_progress`：未授权使用
- 明确的任务成功/失败标志：不存在于 `info` 或 `terminated` 中，只能通过终止条件的名称推断，不能直接用作奖励信号
- 外界风速或其他随机扰动：源码中被省略，不可知
