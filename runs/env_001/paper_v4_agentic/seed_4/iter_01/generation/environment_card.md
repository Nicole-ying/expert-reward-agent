# 匿名环境理解卡片

## 1. 任务目标
任务核心是控制一个具有两条支撑腿的 2D 飞行器（启动时带有随机初始扰动），从视口顶部中央附近出发，安全、平稳地降落到画面中央的水平目标平台上，并稳定停靠。主要目标是抵达目标位置并实现“软着陆”（低速、姿态竖直、支撑腿接触），尽量减少发动机使用量（燃料消耗），同时鼓励快速完成。附属目标为姿态保持、节能及时间效率，但不应与安全降落冲突，也不应被误认为单纯的点对点导航或纯粹的平衡维持任务。

## 2. 任务类型选择
- **selected_route_id**: navigation_goal_reaching
- **confidence**: high
- **reason**: 任务的支配性目标是到达空间中的一个指定目标位置（中心降落平台），这与 navigation_goal_reaching 的定义高度吻合。虽然任务包含节能、稳定等次目标，但它们只是附属优化项，核心仍是目标到达。任务不存在多个权重大致相当且冲突的目标（多目标情况）。该任务在 navigation_goal_reaching 下面进行更细粒度的动力学子类型划分。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: float32（根据 Box 推断）
- **各维含义与 reward_usable 属性**：
  - **obs[0]**: x_position — 水平坐标，相对于目标平台的水平偏移，reward_usable: **true**
  - **obs[1]**: y_position — 垂直坐标，相对于目标平台高度的偏移，reward_usable: **true**
  - **obs[2]**: x_velocity — 水平线速度，reward_usable: **true**
  - **obs[3]**: y_velocity — 垂直线速度，reward_usable: **true**
  - **obs[4]**: body_angle — 机体倾斜角度，reward_usable: **true**
  - **obs[5]**: angular_velocity — 机体角速度，reward_usable: **true**
  - **obs[6]**: left_support_contact — 左支撑腿接触标志（1.0 表示接触，0.0 表示未接触），reward_usable: **true**
  - **obs[7]**: right_support_contact — 右支撑腿接触标志，reward_usable: **true**

所有观测字段均可直接用于奖励计算。

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **具体动作与含义**：
  - **action 0**: no_engine — 不启动任何引擎，自由滑行
  - **action 1**: left_orientation_engine — 启动左侧姿态引擎，产生使机体逆时针（或对应方向）旋转的力矩
  - **action 2**: main_engine — 启动主引擎，产生垂直向上的推力（减速或悬停）
  - **action 3**: right_orientation_engine — 启动右侧姿态引擎，产生与左引擎反向的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**：身体稳定停靠（body_not_awake_or_settled）且至少有一只支撑腿接触地面，且没有发生 crash 或出界。这是期望的成功状态，表现为速度极小、姿态接近竖直、接触信号为 1，但无法从 info 直接读取，必须通过观测信号间接推断。
- **failure-like termination**：
  - crash_or_body_contact：身体主体（非支撑腿）接触地面或其他碰撞导致坠毁，通常与高速、大角度撞击有关。
  - horizontal_position_outside_viewport：水平位置超出可显示边界，即机体飞离有效区域。
- **ambiguous termination**：body_not_awake_or_settled 但左右支撑腿均未接触——可能代表机体已倒地且静止，本质上属于失败。
- **truncation**：未提及显式 step 限制，但可能存在隐式最大步数（环境未披露），此时 info 为空字典，无法直接识别。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false （info 为空字典，无任何成功标志）
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无（info 为空）
- **forbidden_or_uncertain_info_fields**: 所有通常可能存在于 info 中的字段如 "success"、"failure"、"termination_reason"、"reward_components" 等均不存在，且不得假设它们可用。终止条件只能通过观测组合（位置、速度、角度、接触）以及是否在达到稳定/边界时 episode 结束来间接推断，标记为 **derived_possible**。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
- **允许使用的输入**：
  - `obs`：当前步的观测数组，shape=(8,)，可使用全部 8 维。
  - `action`：当前步执行的动作整数（0~3）。
  - `next_obs`：下一时刻的观测数组，可用于评估状态变化、速度变化、是否接触等。
  - `info`：仅当环境明确提供字段时方可使用，本环境 `info` 为空字典，因此禁止使用任何 info 字段。
  - `training_progress`：浮点数 0.0~1.0，仅在 prompt 特别要求或允许时才可使用。
- **禁止使用的输入**：
  - `original_reward`：官方原始奖励已被屏蔽，严禁以任何形式复现或直接使用。
  - 任何未在观测空间中声明的 obs 切片或虚构的 info 字段。
  - 任何假设的成功/失败标志。

## 7. 可用于奖励函数的信号
- **位置信号**：`obs[0] x_position`、`obs[1] y_position`、`next_obs[0]`、`next_obs[1]`。可用于计算到目标 (0,0) 的距离、高度误差等。
- **速度信号**：`obs[2] x_velocity`、`obs[3] y_velocity`、`next_obs` 中对应项。可用于惩罚高速撞击或奖励低速软着陆。
- **姿态信号**：`obs[4] body_angle`、`obs[5] angular_velocity`、`next_obs` 对应项。可用于鼓励竖直姿态和减少旋转。
- **接触信号**：`obs[6] left_support_contact`、`obs[7] right_support_contact`、`next_obs` 对应项。可用于奖励支撑腿接触，表示着陆成功。
- **动作/引擎信号**：`action` 取值可用于计算燃料消耗（若 action ≠ 0 则为引擎启用）。
- **衍生推断信号（derived_possible）**：
  - 邻近成功：当 `next_obs` 中支撑腿接触为 1，且 `next_obs` 的 `x_velocity`、`y_velocity`、`body_angle` 接近 0，`y_position` 接近 0，可推断为成功软着陆。虽然无法从 info 获得标识，但在连续奖励中可通过组合条件给出额外奖励。
  - 坠毁推断：`next_obs` 中 `body_angle` 突然大幅偏离 0 或 `y_position` 突变（被重置），可间接推测崩溃，但不要用于奖励，仅用于诊断。
  - 出界推断：`x_position` 超出合理范围（如 >1 或 <-1），可用于惩罚，但此时环境已终止，一般不需要奖励。

## 8. 不确定或不可用的信号
- 明确的成功/失败布尔标志：info 中无任何字段。
- episode 终止标志：`compute_reward` 签名中未提供 `terminated` 参数，无法获知当前步是否为最后一步。
- 真实燃料量 / 剩余能量：动作空间只有离散引擎选择，不提供连续推力或剩余能量观测。
- 距离目标的绝对航程：无直接航程观测，但可由位观测算得。
- 发动机推力大小：所有引擎的推力强度未观测，只能通过动作类型间接估计。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: lander with two legs
  actuator_type: one main vertical thruster + two lateral orientation thrusters
  contact_structure: two point-like foot contacts (left, right)
primary_objectives:
  - reach the central target pad (minimize position error)
  - achieve soft touchdown (low velocity, low angular rate, upright)
