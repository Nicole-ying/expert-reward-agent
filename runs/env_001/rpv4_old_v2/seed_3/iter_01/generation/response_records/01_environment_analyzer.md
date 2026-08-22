# Response Record

# 匿名环境理解卡片

## 1. 任务目标
一个 2D 飞行器从画面上方中心附近出发，受初始随机力作用。核心目标是**到达并稳定在中心目标平台上**（即精准着陆并停稳）。任务同时要求**尽可能快**地完成着陆，并且**尽可能少使用发动机推力**。在过程中，智能体需要学会接近目标、降低速度、保持姿态稳定并实现安全接触（避免硬撞击）。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: **high**  
reason: 核心目标是到达并稳定在目标平台上，加快速度与节省推力属于附属优化，不存在多目标之间不可调和的主次冲突。动力学子类型符合 **goal_approach_and_soft_contact**，因为需要减速、稳定姿态并安全触地。

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: (8,)  
- **dtype**: float32（推测）  
- **obs[0]**: `x_position`，水平坐标相对于目标 pad，可用于衡量横向距离，reward_usable: true  
- **obs[1]**: `y_position`，垂直坐标相对于 pad 高度，可用于衡量高度，reward_usable: true  
- **obs[2]**: `x_velocity`，水平线速度，用于速度控制，reward_usable: true  
- **obs[3]**: `y_velocity`，垂直线速度，用于软着陆控制，reward_usable: true  
- **obs[4]**: `body_angle`，机体角度，用于姿态稳定性，reward_usable: true  
- **obs[5]**: `angular_velocity`，角速度，用于姿态稳定，reward_usable: true  
- **obs[6]**: `left_support_contact`，左侧支撑接触标志（0/1），用于判断着陆状态，reward_usable: true  
- **obs[7]**: `right_support_contact`，右侧支撑接触标志（0/1），用于判断着陆状态，reward_usable: true  

## 4. 动作空间 action_space
- **type**: Discrete  
- **n**: 4  
- **action 0**: `no_engine`，无发动机动作（什么都不做）  
- **action 1**: `left_orientation_engine`，左姿态发动机，提供侧向推力或扭矩  
- **action 2**: `main_engine`，主发动机，提供向上推力  
- **action 3**: `right_orientation_engine`，右姿态发动机，提供另一侧推力或扭矩  

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` – 可能表示机体不再活跃或已稳定（例如成功着陆后静止），但也可能包含坠落静止等失败情况。**无显式成功标志**。  
- **failure-like termination**:  
  - `crash_or_body_contact` – 机体与地面或障碍物碰撞导致的失败。  
  - `horizontal_position_outside_viewport` – 水平位置超出画面边界，视为失败。  
- **ambiguous termination**: 上述 settled 条件含义模糊，其设计意图是探测着陆稳定，但不能直接作为可靠的成功指标。  
- **truncation**: 未提供任何时间步或步数限制的截断信号。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: **false**  
- `explicit_failure_flag_available`: **false**  
- `allowed_info_fields`: 环境返回空字典，无可用的额外信号。  
- `forbidden_or_uncertain_info_fields`: 所有未声明的 info 字段均不可用，且不应假设存在 `success`、`failure` 或 `termination_reason`。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用**：
- `obs` / `next_obs`（当前和下一步观察）
- `action`（当前动作）
- `info` 中明确允许的字段（当前为空，因此不可用）
- `training_progress` 仅在 prompt 明确允许时使用，当前未允许，故禁止使用

**禁止使用**：
- `original_reward`（官方奖励已被屏蔽）
- 任何未声明的 `info` 字段
- 任何未声明的观测切片（如假设观测包含其他维度）

## 7. 可用于奖励函数的信号
- **position**: `obs[0]`（x_position）、`obs[1]`（y_position），可计算到目标 pad 的距离；当两者趋近于 0 时表示到达。  
- **velocity**: `obs[2]`（x_velocity）、`obs[3]`（y_velocity），可用于惩罚或塑造速度，尤其在接近目标时。  
- **orientation**: `obs[4]`（body_angle）、`obs[5]`（angular_velocity），用于保持竖直与姿态稳定。  
- **contact**: `obs[6]`（left_support_contact）、`obs[7]`（right_support_contact），指示是否已接触环境，用于判断着陆状态与安全性。  
- **action/engine**: `action` 取值 1/2/3 代表发动机点火，可计算燃料消耗代价。  
- **other**: 无其他明确可用的信号。

## 8. 不确定或不可用的信号
- 显式的成功/失败标志：无。  
- 目标 pad 的绝对位置或 viewport 边界：未提供，只能通过相对位置推断。  
- 燃料剩余量：没有直接观测值，只能用动作序列间接估计。  
- 碰撞类型或 crash 细节：终止时无额外信息。  
- 安全垂直速度阈值：未给出，需凭经验设定。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid-body vehicle with landing legs
  actuator_type: two orientation thrusters and one main thruster
  contact_structure: binary leg contacts (left/right)
primary_objectives:
  - reach the central target pad and settle stably
secondary_objectives:
  - minimize time to reach (implied by “as fast as possible”)
  - minimize engine thrust usage
  - maintain stable orientation during descent
main_failure_risks:
  - crashing into ground or obstacles at high velocity
  - exceeding horizontal boundaries
  - failing to stabilize orientation, leading to tilting and instability
  - excessive fuel consumption leading to inability to land softly
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `goal_proximity`  
  **purpose**: 鼓励向目标 pad 移动，最终到达目标位置（x→0, y→0）。  
  **why_required**: 任务核心是到达，距离减小是主要驱动信号。  
  **usable_signals**: [obs[0], obs[1]]  
  **risks**: 仅追求近距离可能鼓励高速俯冲导致 crash，需配合速度与接触约束。

- **role_id**: `velocity_damping`  
  **purpose**: 在接近目标时抑制速度，实现软着陆。  
  **why_required**: 到达目标时必须几乎零速度才能安全稳定，防止硬碰撞。  
  **usable_signals**: [obs[2], obs[3], obs[0], obs[1]]（可利用距离调制惩罚强度）  
  **risks**: 过强速度惩罚阻碍必要机动，推荐随距离衰减。

- **role_id**: `fuel_efficiency`  
  **purpose**: 惩罚使用任何发动机推力，鼓励节省燃料。  
  **why_required**: 任务明确要求“using as little engine thrust as possible”。  
  **usable_signals**: [action == 1, action == 2, action == 3]  
  **risks**: 过度节省可能导致无法减速或姿态失控，需平衡权重。

- **role_id**: `safe_contact`  
  **purpose**: 在触地时确保垂直速度很小，并且双足同时触地更稳定。  
  **why_required**: 任务要求“make safe contact”，避免硬着陆 crash。  
  **usable_signals**: [obs[6], obs[7], obs[3], obs[1]]（接触标志 + y_velocity，可结合高度）  
  **risks**: 若在非目标区域过早激活，可能误导智能体提前接触错误表面，应仅在接近目标高度时启用。

- **role_id**: `orientation_stability`  
  **purpose**: 保持机体竖直，减小旋转。  
  **why_required**: 描述中要求“keep a stable orientation”，且姿态稳定有助于安全接触。  
  **usable_signals**: [obs[4], obs[5]]  
  **risks**: 与机动性有轻微冲突，但权重较低可接受。

### 10.2 条件职责 conditional_roles
- **role_id**: `fast_progress_encouragement`  
  **condition_to_use**: 早期训练可能进展缓慢，可在智能体向目标移动时给予额外正奖励，避免摸鱼。  
  **usable_signals**: [obs[0], obs[1]] 的变化量（通过 next_obs 差分）  
  **risks**: 可能加剧 crash 风险，需与其他安全职责一起使用。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `boundary_penalty`  
  **reason**: 环境已通过终止条件处理越界，边界绝对坐标未知，额外惩罚可能干扰探索。  
  **forbidden_or_missing_signals**: 边界坐标。

- **role_id**: `explicit_success_reward`  
  **reason**: 无明确的 success flag 可用。

- **role_id**: `time_penalty_per_step`  
  **reason**: 每步直接惩罚时间可能与 fuel_efficiency 高度耦合，且过于粗暴。更推荐通过 progress shaping 实现快速接近。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| `goal_proximity` | x_position, y_position | target distance metric | quadratic_penalty, l2_norm | 可用 -sqrt(x²+y²) 或 -(x²+y²) |
| `velocity_damping` | x_velocity, y_velocity, (distance proxy) | desired zero velocity | weighted_squared_penalty, proximity_gated | 速度惩罚乘以随距离衰减的系数，靠近目标时加重 |
| `fuel_efficiency` | action index (1,2,3) | fuel level | action_cost (constant penalty per activation) | 每次点火给予固定负奖励 |
| `safe_contact` | left_contact, right_contact, y_velocity, (y_position) | safe velocity threshold | conditional_penalty, indicator * penalty | 当接触标志为真且高度接近 0 时，惩罚 y_velocity² |
| `orientation_stability` | body_angle, angular_velocity | desired angle=0 | quadratic_penalty | angle² + w·angular_velocity² |
| `fast_progress` (conditional) | (x_position, y_position) diff | – | positive_shaping_on_negative_delta | dist_prev – dist_curr > 0 时给予正奖励 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高速撞地导致早期 crash | 奖励不收敛，episode 极短，频繁 `crash_or_body_contact` | 增加 `velocity_damping` 和 `safe_contact` 权重，强化姿态稳定 |
| 飘出水平边界 | 因越界终止，累计奖励低 | 谨慎增加 soft boundary 惩罚或加强姿态/速度引导 |
| 过度悬停或长时间不下降 | episode 长但距离不减小，燃料惩罚高 | 引入 `fast_progress` 奖励，适当放宽燃料惩罚以允许探索 |
| 单脚着陆导致倾斜失败 | 最终仅一侧接触，body_angle 大 | 在 `safe_contact` 中奖励双接触，对单脚接触施加小负奖励 |
| 姿态剧烈振荡 | angular_velocity 长期大，body_angle 波动 | 提高 `orientation_stability` 权重，或惩罚连续来回点火 |
| 早期耗尽燃料，后期无法减速 | 前期大量使用 main_engine，后期 crash | 加强燃料惩罚，或调整探索噪声鼓励少用主发动机 |
| 无法稳定静止，反复弹跳 | 目标附近仍持续移动 | 在 Y 接近 0 且接触时大幅强化 `velocity_damping`，可配合接触锁死奖励 |

通过观测上述信号调整各职责权重，可在不依赖真实 success flag 的条件下逐步提升智能体性能。
