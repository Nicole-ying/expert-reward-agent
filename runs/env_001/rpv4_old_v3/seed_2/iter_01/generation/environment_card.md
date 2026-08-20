# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 的飞行器/着陆器轨迹优化任务。智能体从一个靠近顶部中央的位置出发，带有随机初始作用力。  
**主目标**：尽快到达画面中央的目标平台，并安全停靠（soft landing），即要求位置收敛到平台上、速度很低、姿态竖直、支撑脚安全触地。  
**次目标**：在完成任务的前提下，尽量少使用引擎推力（节省燃料），同时保持姿态稳定。  
**不应混淆的目标**：该任务不是纯生存或平衡任务——到达目标平台是刚性的成功条件，能量节省只是辅助约束，不是多目标的平等折衷。

## 2. 任务类型选择
**selected_route_id**: `navigation_goal_reaching`  
**confidence**: high  
**reason**: 任务的核心驱动力是移动到指定目标位置（中心平台）并有条件地稳定在那儿。省燃料、快速等只是附加要求，不构成多目标冲突，因此不属于 multi_objective_task。环境没有强调持续行进、驾驶安全或探索稀疏事件，因此 navigation_goal_reaching 的匹配度最高。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: [8]
- **dtype**: float32 (推断)
- **obs[0]**: `x_position`，水平坐标（相对于目标平台中心），用于衡量横向偏移，奖励可用 ✅
- **obs[1]**: `y_position`，垂直坐标（相对于平台高度），用于衡量高度/下降进程，奖励可用 ✅
- **obs[2]**: `x_velocity`，水平线速度，用于减速奖励/惩罚，奖励可用 ✅
- **obs[3]**: `y_velocity`，垂直线速度，用于减速奖励/惩罚，奖励可用 ✅
- **obs[4]**: `body_angle`，机体角度（可能以弧度为单位，0 表示竖直），奖励可用 ✅
- **obs[5]**: `angular_velocity`，角速度，用于稳定姿态，奖励可用 ✅
- **obs[6]**: `left_support_contact`，左支撑脚接触标志（0/1），指示是否触地，奖励可用 ✅
- **obs[7]**: `right_support_contact`，右支撑脚接触标志（0/1），指示是否触地，奖励可用 ✅

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **动作含义**：
  - `0`: `no_engine` – 不激活任何引擎，仅靠惯性飞行。
  - `1`: `left_orientation_engine` – 点火左定向引擎，产生一个使机体逆时针（或某一方向）旋转的力矩。
  - `2`: `main_engine` – 点火主引擎，产生向上的推力（对抗重力并控制下降速度）。
  - `3`: `right_orientation_engine` – 点火右定向引擎，产生与动作 1 相反的旋转力矩。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `MASKED_STEP_SOURCE` 的 `terminated` 计算逻辑：
- **crash_or_body_contact**：躯体发生碰撞或身体支撑点接触地面。此情景可能包括：
  - 成功的软着陆（接触时速度极低、姿态正）。
  - 失败的高速撞击/翻倒等。
- **horizontal_position_outside_viewport**：x 坐标超出视区边界，明确失败。
- **body_not_awake_or_settled**：躯体进入睡眠状态或已稳定（settled）。这可能是：
  - 成功着陆后自然睡眠。
  - 坠毁后物理引擎判定身体安定，属于失败。

**总结终止模式**：
- **success-like termination**：`body_not_awake_or_settled` 在速度接近于 0、位置接近目标平台、接触标志为 1 且角度接近 0 时发生。
- **failure-like termination**：`horizontal_position_outside_viewport`；或因高速、大角度导致的 `crash_or_body_contact`/`body_not_awake_or_settled`。
- **ambiguous termination**：`crash_or_body_contact`/`body_not_awake_or_settled` 需结合 next_obs 判定成败。
- **truncation**：未提及，但有最大 episode 长度（通常存在），应在 reward 中注意时间步惩罚。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
  （`info` 字典为空，无任何显式标志）
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无
- **forbidden_or_uncertain_info_fields**: 所有 info 字段（因为没有提供）

> 因此，任何成功/失败的判断必须从 `next_obs` 与 `terminated` 的组合中推断，而不能依赖 `info`。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

- **允许使用**：
  - `obs`
  - `action`
  - `next_obs`
  - `info` 中的字段：无（info 为空，使用它没有任何实际效果）
  - `training_progress`：仅在 prompt 明确允许时使用，此处未明确允许，**建议禁止使用**。
- **禁止使用**：
  - `original_reward`（被屏蔽的官方奖励）
  - 任何未声明的 `info` 字段
  - 在 `obs` 中假设的未列出维度

## 7. 可用于奖励函数的信号
- **position**：`x_position`, `y_position`（相对目标），可计算到目标的欧氏距离或分别控制。
- **velocity**：`x_velocity`, `y_velocity`，可设计近地减速、稳定约束。
- **orientation**：`body_angle`，鼓励竖直。
- **angular**：`angular_velocity`，鼓励旋转抑制。
- **contact**：`left_support_contact`, `right_support_contact`，标志着陆接触已发生，可配合速度用于判定软着陆。
- **action/engine**：`action` 选择，可用于惩罚引擎使用（燃料节省）。
- **other**：`terminated` 标志（作为环境返回，但 `compute_reward` 接口未直接提供，可通过在调用 `compute_reward` 时传入 `done` 标志获取；我们可以在训练循环中以额外参数传递，但接口中只有 `next_obs` 和 `info`；`terminated` 通常可在调用时通过另一个变量获得，若必须在 `compute_reward` 中获取，需要从外部传入——在此假设可以在调用时提供 `done` 标志，否则可通过 `next_obs` 的特征推断终止状态。由于题干未明确禁止，但谨慎起见，直接使用 `done` 需要额外参数，在标准接口中不鼓励，所以这里只强调可从 `next_obs` 和 `info` 推断。但 info 空，所以成功稀疏奖励只能基于 `next_obs` 的状态是否符合成功条件给出，而不依赖外部 `done`。所以我们仍可以将“是否处于终止且状态符合成功条件”作为奖励计算的一部分，但必须在训练循环的每个 step 结束时调用，需要能够判断 `done`，这部分由训练框架负责。我们此处声明信号可用，但不具体到实现细节。）

## 8. 不确定或不可用的信号
- `original_reward`：不可用。
- 任何 `info` 字段：不可用（因为 info 为 `{}`）。
- 明确的成功/失败标志：不存在，不能直接用作奖励源。
- 世界坐标系绝对位置：未知，只有相对位置。
- 引擎推力强度、燃料剩余量：未提供，不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete (action space) acting on continuous dynamics
morphology:
  body_type: 矩形或近似多边形机体，底部有两个支撑脚
  actuator_type: 三个引擎（一个主推力引擎，两个定向旋转引擎）
  contact_structure: 两个底部支撑点（left/right），独立检测接触
primary_objectives:
  - 到达目标平台中心位置 (x≈0, y≈0)
  - 实现安全软着陆（低速度、角度竖直、支撑脚稳定接触）
secondary_objectives:
  - 最小化引擎推力使用（节约燃料）
  - 快速完成任务（尽快）
main_failure_risks:
  - 高速撞击平台导致坠毁
  - 姿态失控旋转
  - 飞出水平边界
  - 长时间无法收敛而耗尽训练步数（truncation）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `goal_progress_guidance`  
  **purpose**: 持续引导智能体靠近目标平台，使之尽快到达目标水平位置和高度。  
  **why_required**: 这是核心目标，没有此引导，agent 可能随机游走，无法稳定达到目标。  
  **usable_signals**: `x_position`, `y_position`, `x_velocity`, `y_velocity`  
  **risks**: 单纯基于位置的奖励可能鼓励 agent 高速冲向目标，忽略减速，导致坠毁。

- **role_id**: `soft_landing_safety`  
  **purpose**: 在接近平台时严格要求低速、竖直姿态，防止坠毁。  
  **why_required**: 任务要求“安全接触”，即着陆时损失小、姿态正；否则即使位置到达也算失败。  
  **usable_signals**: `y_velocity`, `x_velocity`, `body_angle`, `angular_velocity`, `left_support_contact`, `right_support_contact`  
  **risks**: 过于严格的惩罚可能使 agent 不敢下降或长时间悬停不触地。

- **role_id**: `energy_efficiency`  
  **purpose**: 鼓励少用主引擎和定向引擎，降低燃料消耗。  
  **why_required**: 任务描述明确要求“as little engine thrust as possible”。  
  **usable_signals**: `action` (特别是 `main_engine` 和两个定向引擎的选择)  
  **risks**: 权重过高会导致 agent 放弃使用引擎，无法克服重力下降，完不成主任务。

### 10.2 条件职责 conditional_roles
- **role_id**: `horizontal_alignment_and_stabilization`  
  **condition_to_use**: 当智能体 y 坐标已经接近 0（已接近着陆高度）或垂直速度已经很低时，加强水平位置和角度的稳定奖励。  
  **usable_signals**: `x_position`, `x_velocity`, `body_angle`, `angular_velocity`  
  **risks**: 过早强化可能干扰飞行轨迹的灵活性。

- **role_id**: `sparse_mission_success_reward`  
  **condition_to_use**: 仅在成功着陆时刻（基于 next_obs 判断满足位置、接触、速度、角度等所有成功条件）提供一次较大的奖励。  
  **usable_signals**: 全部 next_obs 字段，不可依赖 info。  
  **risks**: 稀疏信号可能难以学习，需要与密集奖励共同使用；若定义不严格，可能将失败态误判为成功。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `survival_balance`  
  **reason**: 任务没有持续的平衡生存危机（如倒立摆、独轮车），主目标不是维持存活，而是到达目标。  
  **forbidden_or_missing_signals**: 不具备必要的“生存时间”或“活过多少步”的显式信号。
- **role_id**: `multi_objective_trade_off`  
  **reason**: 虽然有多项指标，但到达并安全着陆是绝对优先，燃料节省只是次要目标，不需要复杂的帕累托折衷，因此不属于 multi_objective_task 框架中的平等多目标。
- **role_id**: `exploration_bonus`  
  **reason**: 环境状态空间相对简单，目标明确，不需要刻意鼓励状态空间探索；盲目添加可能会噪声干扰学习。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_progress_guidance | x_position, y_position | none | distance_penalty (L2), bounded_reward (per‑dimension), potential‑based shaping | 鼓励同时减小水平与垂直偏差 |
| soft_landing_safety | y_velocity, x_velocity, body_angle, angular_velocity, left_support_contact, right_support_contact | 无 | velocity_penalty (clipped), angular_penalty, contact_conditional (仅在接触时加强速度/角度惩罚) | 需要在下降末段激活 |
| energy_efficiency | action (离散值) | 无 | action_cost (对 action≠0 惩罚，或对1,2,3分别加权) | 可与进度耦合，避免“躺平” |
| horizontal_alignment_and_stabilization | x_position, x_velocity, body_angle, angular_velocity | 无 | quadratic_penalty, conditional_scale (在 y 接近 0 时放大) | 增加着陆横向稳定性 |
| sparse_mission_success_reward | 全部 next_obs | 无 | success_flag (布尔逻辑) → fixed_bonus | 必须严格定义成功门限，避免 reward hacking |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 不动或轻微漂浮，不下降 | 动作全程为 0 或主引擎极少点火；y_position 几乎不变 | 增加 goal_progress_guidance 权重，考虑基于高度的势能奖励，或降低能量惩罚 |
| 高速冲向平台坠毁 | y_velocity 大幅为负，接触时速度大 | 强化 soft_landing_safety 在低高度的速度惩罚，或使用速度上限饱和惩罚 |
| 在目标上空严重摇摆或旋转 | body_angle 或 angular_velocity 震荡 | 加强 orientation_convergence 的惩罚权重，或添加角速度阻尼奖励 |
| 飞出水平边界被截断 | x_position 绝对值快速增加 | 增加横向偏离的惩罚，或引入边界激活的惩罚（如 \( \|x\| \) 过大时大幅惩罚） |
| 成功悬停但迟迟不着陆 | y_position 接近零但一直不接触，持续消耗能量 | 在高度很低时引入“着陆鼓励”附加奖励（contact 瞬间奖励）或降低主引擎奖励，或使能量惩罚与时间成比例增加 |
| 稳定着陆后仍继续引擎点火 | 着陆后接触 = 1，但仍选择动作 1,2,3 | 当着陆接触满足速度条件时，加大 action_cost，并给出成功稀疏奖励后截断，或抑制引擎使用 |