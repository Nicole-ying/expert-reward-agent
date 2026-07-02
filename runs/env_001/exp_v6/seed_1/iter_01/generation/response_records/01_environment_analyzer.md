# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
在一个 2D 平面中，控制一辆类似飞行器/小车的智能体从上方起始位置出发，尽快到达并稳定悬停于中心目标平台（target pad）上。要求使用尽可能少的引擎推力（能量），同时保持姿态稳定、与平台安全接触。智能体需要学会精确抵近、减速、维持姿态并最终停靠。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 环境明确给出了一个固定的目标位置（中心目标平台），核心需求是高效抵达并稳定在目标处，属于典型的目标导向导航任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推断为 float32（位置、速度、角度为浮点数，接触标志用 0.0/1.0 表示）
- obs[0]: x_position —— 智能体质心在水平方向上相对于目标平台中心的距离
- obs[1]: y_position —— 智能体质心在垂直方向上相对于平台高度的距离
- obs[2]: x_velocity —— 水平方向线速度
- obs[3]: y_velocity —— 垂直方向线速度
- obs[4]: body_angle —— 机体朝向角（弧度/角度，具体量纲未公开）
- obs[5]: angular_velocity —— 角速度
- obs[6]: left_support_contact —— 左支撑点/起落架是否与平台接触（0.0 表示未接触，1.0 表示接触）
- obs[7]: right_support_contact —— 右支撑点/起落架是否与平台接触（同上）

## 4. 动作空间 action_space
- type: Discrete
- 动作数量: 4
- action 0: no_engine —— 不施加任何引擎推力，仅靠惯性运动
- action 1: left_orientation_engine —— 点燃左侧姿态调节引擎
- action 2: main_engine —— 点燃主引擎（通常产生向上的推力）
- action 3: right_orientation_engine —— 点燃右侧姿态调节引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - `body_not_awake_or_settled`：智能体不再被唤醒或已经稳定停靠（可能表示已安静停在目标平台上）。
- failure-like termination:
  - `crash_or_body_contact`：发生碰撞或机体与地面/障碍物不当接触（非目标平台的安全接触）。
  - `horizontal_position_outside_viewport`：水平位置超出视口边界。
- ambiguous termination:
  - 无。
- truncation:
  - 本文档中未出现时间步限制截断，step 返回的 truncated 恒为 False。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（step 返回的 info 为空字典 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；原始环境没有提供任何 success/failure 标志位

注意：奖励函数只能从（原始）观察和动作中推断成败，不能依赖 info 中的任何键。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs —— 当前步骤之前的观察向量（shape (8,)）
- action —— 当前步骤执行的动作（0-3）
- next_obs —— 动作执行后的新观察向量（shape (8,)）
- info —— 步骤返回的 info 字典，但本环境中 info 为空 {}，不能从中提取信息
- training_progress —— 训练进度（0 到 1 之间的浮点数），**仅在 prompt 明确允许时使用，本描述未提及可用，故默认禁止**

禁止使用：
- original_reward —— 环境原始奖励（已遮蔽，不可获取）
- official_reward —— 同上
- 未在观察空间中声明的 obs 切片/维度
- info 中的任何字段（因为 info 为空）

## 7. 可用于奖励函数的信号
- position: obs[0] (x), obs[1] (y) 以及 next_obs[0], next_obs[1] —— 可用于设计距离引导、目标抵达奖励
- velocity: obs[2] (vx), obs[3] (vy) 以及 next_obs[2], next_obs[3] —— 可用于惩罚接近目标时的高速度，鼓励稳定
- orientation: obs[4] (angle) 以及 next_obs[4] —— 鼓励保持水平/目标姿态
- contact: obs[6], obs[7] 以及 next_obs[6], next_obs[7] —— 可用于检测是否双脚平稳接触平台（安全着陆）
- action/engine: action —— 可用于惩罚引擎使用（推力成本）

## 8. 不确定或不可用的信号
- 没有明确的成功/失败标签（info 中无任何键值）
- 没有环境返回的“续航步数”或“接近目标”的显式指标
- 没有目标位置之外的额外导航信息（如绝对坐标、边界位置等）
- training_progress 未获得明确使用许可，不应作为奖励输入
- 原始的官方奖励被完全遮蔽，不能依赖或猜测其形式
