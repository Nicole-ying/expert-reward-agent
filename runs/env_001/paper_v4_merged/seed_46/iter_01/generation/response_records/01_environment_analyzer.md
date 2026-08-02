# Response Record

# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从画面顶部附近出发，尽快飞到中央目标平台并稳定着陆，同时尽量减少引擎推力使用。  
次优要求：保持姿态稳定、安全轻触平台、避免侧向偏移过大。  
不应混淆的目标：单纯存活（有明确的到达位置要求），也不是无限制前进（目标是一个固定点位）。

## 2. 任务类型选择
selected_route_id: `navigation_goal_reaching`  
confidence: `high`  
reason: 任务核心是带着初始随机力、以尽可能快的速度到达固定目标点并驻留，这属于典型的导航目标到达问题。燃料节省和轻柔着陆是附属优化，不影响任务主要类别。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `(8,)`
- dtype: `float32`（假定，实际由环境决定）
- obs[0]: `x_position` – 相对于目标平台的水平距离，`reward_usable: true`
- obs[1]: `y_position` – 相对于平台高度的垂直距离，`reward_usable: true`
- obs[2]: `x_velocity` – 水平线速度，`reward_usable: true`
- obs[3]: `y_velocity` – 垂直线速度，`reward_usable: true`
- obs[4]: `body_angle` – 机体倾角，`reward_usable: true`
- obs[5]: `angular_velocity` – 角速度，`reward_usable: true`
- obs[6]: `left_support_contact` – 左支撑触地标志（0/1 或连续），`reward_usable: true`
- obs[7]: `right_support_contact` – 右支撑触地标志，`reward_usable: true`

## 4. 动作空间 action_space
- type: `Discrete`
- n: `4`
- action 0: `no_engine` – 无推力，惯性飞行
- action 1: `left_orientation_engine` – 启动左姿态发动机（主要用于调整角速度）
- action 2: `main_engine` – 启动主发动机（提供反推力/升力）
- action 3: `right_orientation_engine` – 启动右姿态发动机（与左姿态对称）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` – 当机体静止（可能已着陆并稳定）时触发。此模式极可能表示成功着陆，尤其是配合近距离、低速、良好姿态和接触信号。
- **failure-like termination**: `crash_or_body_contact` – 机体与地面或其他物体非腿部接触（推测会导致姿态破坏、超出限制等）。  
  `horizontal_position_outside_viewport` – 水平位置超出视野，必然失败（飞离目标区）。
- **ambiguous termination**: 当同时满足多个条件时（如 crash 且出界），仍视为失败；但仅靠观测无法区分触发原因。`crash_or_body_contact` 和 `body_not_awake_or_settled` 可能同时触发，需以 fail 为准（因为 crash 优先级高）。
- **truncation**: 源码中未看到最大步数截断，但实际使用中可能存在。无额外截断标志。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: `false` (info 为空)
- `explicit_failure_flag_available`: `false`
- `allowed_info_fields`: 无（info 为 `{}`）
- `forbidden_or_uncertain_info_fields`: 任何假设的 `info["success"]`、`info["termination_reason"]` 等均禁止使用

即使没有明确标志，成功可通过终止后状态间接推断：位置接近零、速度极小、姿态平直且至少有一只脚接触平台。这属于 `derived_possible` 信号，在奖励设计时必须谨慎。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs` – 当前状态 (8,)
- `action` – 本次动作 (离散 0‑3)
- `next_obs` – 下一步状态 (8,)
- `info` 必须为空 `{}`，不读取任何字段
- `training_progress` 仅在 prompt 明确允许时使用（此处未允许，禁止）

禁止使用：
- `original_reward`
- 任何未声明的 `info` 字段
- 任何未声明的 `obs` 切片（例如假设存在额外维度）

## 7. 可用于奖励函数的信号
- **position**: `next_obs[0]` (x 偏差), `next_obs[1]` (y 偏差) → 可计算到目标的距离
- **velocity**: `next_obs[2]`, `next_obs[3]` → 总速度或分速度
- **orientation**: `next_obs[4]` (机体倾角), `next_obs[5]` (角速度)
- **contact**: `next_obs[6]`, `next_obs[7]` – 左右支撑是否触地
- **action/engine**: 动作选择本身（离散 0‑3）可用于惩罚引擎使用
- **other**:
  - 距离变化量：`delta_distance = distance(obs) - distance(next_obs)`（进步信号）
  - 终端推断成功（derived_possible）：在 episode 结束时，结合位置、速度、倾角、接触判断是否为成功着陆，可给予稀疏终端奖励
  - 终端推断失败（derived_possible）：推断 crash/出界，可给予惩罚（谨慎使用）

## 8. 不确定或不可用的信号
- 真实的 `success` / `failure` 标志
- 环境内置的 reward（被屏蔽）
- 机体是否与地面非腿部接触（仅从终止条件推断）
- 燃料余量、推力大小、发动机当前状态（无直接测量）
- 任何地形信息、风速等未暴露的物理量

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D vehicle/lander with two support legs
  actuator_type: discrete thrusters (main + two orientation)
  contact_structure: two support leg contacts (left and right)
primary_objectives:
  - approach the target pad and minimize position error
  - achieve zero or near-zero velocity at the moment of contact
secondary_objectives:
  - maintain upright orientation (small angle)
  - secure landing with at least one support contact
  - minimize engine usage
main_failure_risks:
  - overshooting the target due to excess velocity
  - crashing or touching non-landing-gear parts
  - drifting horizontally out of viewport
  - excessive oscillation leading to unstable orientation at landing
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `goal_proximity_progress`
  - purpose: 鼓励每一步向目标靠近，使用距离的减少量作为主信号
  - why_required: 导航任务必须驱动 agent 接近目标。使用 `delta(distance)` 而非绝对距离，避免 agent 在远处悬停得分。
  - usable_signals: `obs[0]`, `obs[1]`, `next_obs[0]`, `next_obs[1]`
  - risks: 若环境存在大跳跃或数值不稳定，delta 可能被固有噪声淹没；需要截断或缩放

- **role_id**: `landing_gentleness`
  - purpose: 在非常接近目标（如距离 < 阈值）且速度较大时惩罚，或在临近着陆时奖励低速度
  - why_required: 任务要求 “reduce velocity” 和 “safe contact”，单靠位置进步无法抑制高速冲击
  - usable_signals: `next_obs[2]`, `next_obs[3]`, 距离阈值
  - risks: 可能过早惩罚速度导致 agent 不敢接近，需要只在接近目标时激活

### 10.2 条件职责 conditional_roles
- **role_id**: `terminal_success_bonus` (derived_possible)
  - condition_to_use: episode 结束且通过状态推断成功着陆（距离 < 阈值、速度 < 阈值、倾角 < 阈值、至少一足触地）
  - usable_signals: `next_obs[0]`, `next_obs[1]`, `next_obs[2]`, `next_obs[3]`, `next_obs[4]`, `next_obs[6]`, `next_obs[7]`
  - risks: 由于无明确成功标志，推断可能出错（如 crash 后静止在目标附近）。建议约束条件比较严格且 bonus 适中，避免成为唯一驱动力

- **role_id**: `orientation_penalty`
  - condition_to_use: 在全程可用，但一般只在倾角超出安全区间（如 |angle| > 0.3 rad）时才施加惩罚（hinge form）
  - usable_signals: `next_obs[4]`
  - risks: 倾角惩罚若过于严格会干扰 exploration；采用 hinge 不惩罚小角度，避免与距离奖励冲突

- **role_id**: `engine_efficiency_penalty`
  - condition_to_use: 任务明确要求 “using as little engine thrust as possible”，因此可对主引擎和姿态引擎的使用施以轻微惩罚，但权重应远低于主目标
  - usable_signals: `action` (0‑3)
  - risks: 过重的燃料惩罚可能使 agent 学会不发动引擎、无法到达目标，需极小权重

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `survival_only_time`
  - reason: 本环境有明确的到达目标要求，存活时间奖励会导致 agent 在远离目标处盘旋，与主要目标冲突
  - forbidden_or_missing_signals: 无作用，但生存奖励在本任务中应禁用

- **role_id**: `permanent_velocity_penalty`
  - reason: 全程惩罚速度会与接近目标所需的机动矛盾，只能在接近阶段使用 velocity penalty；全程惩罚会抑制快速移动，违背 “as fast as possible” 的次优化
  - forbidden_or_missing_signals: 无

- **role_id**: `full_body_contact_penalty`
  - reason: 环境未提供非腿部接触的观测信号，无法直接惩罚。可通过终止推断失败，但不宜作为密集惩罚
  - forbidden_or_missing_signals: 缺少 `body_contact` 传感器

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| `goal_proximity_progress` | `obs[0], obs[1], next_obs[0], next_obs[1]` | none | `delta(distance)`, `bounded_signal` | 主信号，需限制单步最大变化，防止悬停 reward 截断 |
| `landing_gentleness` | `next_obs[2], next_obs[3]`, distance | none | `hinge_penalty` (只在距离小于阈值且速度大于阈值时激活) | 属于进展辅助，可与 `terminal_success_bonus` 协同 |
| `terminal_success_bonus` (derived_possible) | `next_obs[0], next_obs[1], next_obs[2], next_obs[3], next_obs[4], next_obs[6], next_obs[7]` | 无明确 success flag | `sparse_bonus` | 仅当 episode 结束且满足多条件时触发 |
| `orientation_penalty` | `next_obs[4]` | none | `hinge_penalty` | 角度在安全区内不惩罚，超出线性惩罚 |
| `engine_efficiency_penalty` | `action` | 推力大小 | `multiplicative_penalty` (与动作选择相关) | 推荐极低权重，如 `-0.01 * (action != 0)` |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停在远离目标处 | reward 平稳但不增加，主信号 `delta` 接近零，agent 位置偏差大 | 检查 `delta` 奖励是否有上限饱和；可能需要降低常数偏移或使用更敏感的缩放 |
| 高速撞击下落区域 | 终端状态速度很高，agent 经常 crash 终止 | `landing_gentleness` 启动过早或权重过小，或 `terminal_success_bonus` 推断条件过松 |
| 不敢启动引擎 | 动作多为 0，进展极慢，agent 几乎不移动 | 燃料惩罚过重，应减小 `engine_efficiency_penalty` 权重，或在初期阶段暂时关闭 |
| 反复横跳导致倾斜 | 倾角持续振荡，角速度大，最后侧向掉出视野 | `orientation_penalty` 可能需要降低安全角度阈值，或增加对角速度的阻尼惩罚 |
| 成功着陆误判为 crash 惩罚 | 推断失败时给予负奖励，但实际着陆成功 | 重新审视 terminal 推断的条件集合，放宽速度/倾角阈值，或暂时移除失败惩罚，只用正向 success bonus 引导 |
