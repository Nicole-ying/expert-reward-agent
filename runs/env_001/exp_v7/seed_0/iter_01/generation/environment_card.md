# Env_001 环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从视野顶部中心出发，通过离散推力指令，以**尽可能短的时间**安全到达并稳定停在中央目标垫上，同时**尽可能少地消耗发动机燃料**。  
需要学会：快速靠近目标、降低速度、保持姿态稳定并实现安全接触。任务同时优化 **移动速度** 与 **燃料经济性**，属于典型的多目标权衡问题。

## 2. 任务类型选择
selected_route_id: multi_objective_task  
confidence: high  
reason: 任务描述明确要求同时优化“到达目标的速度”（时间）和“燃料消耗”（最少推力），是典型的双目标权衡问题，不属于单纯的导航、运动控制或抓取，故归类为多目标任务。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: float32（推测，连续量通常为 float32；contact 信号为 0.0 或 1.0）
- obs[0] (x_position): 飞行器水平坐标相对于目标垫中心的偏移量
- obs[1] (y_position): 飞行器垂直坐标相对于目标垫高度的偏移量
- obs[2] (x_velocity): 水平速度
- obs[3] (y_velocity): 垂直速度
- obs[4] (body_angle): 飞行器当前朝向角
- obs[5] (angular_velocity): 角速度
- obs[6] (left_support_contact): 左侧支撑脚/传感器是否与目标垫接触（1.0=接触, 0.0=未接触）
- obs[7] (right_support_contact): 右侧支撑脚/传感器是否与目标垫接触（1.0=接触, 0.0=未接触）

## 4. 动作空间 action_space
- type: Discrete, n=4
- action 0: 无引擎（不产生任何推力）
- action 1: 左姿态引擎（点火左侧姿态调整发动机，产生绕质心的力矩或侧向推力）
- action 2: 主引擎（点火主推力发动机，通常产生主要向上/向后的推力）
- action 3: 右姿态引擎（点火右侧姿态调整发动机，作用与左姿态引擎相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  飞行器到达目标垫附近，速度极低、姿态稳定、双支撑脚均接触（obs[6] 和 obs[7] 均为 1.0），且不再主动移动时，环境会因 `body_not_awake_or_settled` 而终止。这一条件可视为成功软着陆。
- failure-like termination:  
  - `crash_or_body_contact`：飞行器主体与地面、目标垫边缘或其他障碍物发生非支撑脚接触，代表坠毁。  
  - `horizontal_position_outside_viewport`：水平位置超出允许范围（掉出屏幕），代表失败。
- ambiguous termination:  
  - `body_not_awake_or_settled` 本身不区分成功与失败，若飞行器在错误位置或姿态下静止同样会触发该条件。必须结合位置、速度、接触情况才能判断成功与否。
- truncation: 本环境 step 返回的 truncated 恒为 False（`return ..., False, {}`），无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 为空字典）  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；不能假设存在 success、termination_reason 等键。

**结论**：只能从观测值与终止标志的组合自行推断成功/失败，无法直接读取环境内置的成功标识。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs` (当前观测，一维数组长度为8)
- `action` (当前执行的动作编号)
- `next_obs` (执行动作后的观测)
- `info` 中**不存在**任何允许字段，禁止使用
- `training_progress` 在本次 prompt 未明确允许使用，禁止用于奖励

禁止使用：
- `original_reward` （正式环境 reward 已屏蔽）
- 任何名为 official_reward 或封装的原始奖励信号
- 未在上述允许列表中的 info 字段
- 超范围的 obs 切片（仅可使用 0~7 索引）

## 7. 可用于奖励函数的信号
以下信号均可安全用于奖励定义：
- position: `obs[0]`, `obs[1]`（目标垫相对坐标）
- velocity: `obs[2]`, `obs[3]`
- orientation: `obs[4]`, `obs[5]`
- contact: `obs[6]`, `obs[7]`
- action: `action` 值（当前动作编号，可用于燃料惩罚）
- next_obs 中对应的上述索引同样可用（如 `next_obs[0]` 等）
- 可以计算的辅助量：到目标垫的距离、速度幅值、姿态角误差等。

## 8. 不确定或不可用的信号
- 明确的成功/失败标志：不存在，info 为空，不能依赖。
- 环境内部奖励：已被屏蔽，禁止使用。
- 接触之外的力学信息（如冲击力、滑动摩擦等）：观测中未提供，不可用。
- 任何未在观察空间中声明的维度。