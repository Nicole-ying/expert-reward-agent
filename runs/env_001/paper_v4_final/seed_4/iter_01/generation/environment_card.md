# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 轨迹优化与精准着陆任务。一架拥有主引擎和双定向引擎的飞行器从视口顶部中心附近随机初始力释放，需要尽快飞到中心目标垫（着陆平台）上，实现安全、稳定的接触着陆。主要目标是**快速到达并悬停/降落在目标垫上**，次要目标是**节约燃料（减少引擎使用）**，同时在接近过程中保持姿态稳定，避免猛烈碰撞或飞出视口。任务不应被误解为持续前进或单纯生存平衡：它是以到达指定位置并“软接触”为核心的导航目标到达任务。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**
confidence: high
reason: 核心目标是使飞行器到达中心目标垫并稳定停靠，有明显的到达终点（goal）概念。附属的能量效率、姿态平滑只是围绕主目标的优化。动力学子类型涉及接近目标并低速、稳定接触，符合 goal_approach_and_soft_contact。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（根据典型实现推断）
- **obs[0]**: `x_position` – 飞行器相对于目标垫的水平坐标（偏移），reward_usable: true （可用作距离度量）
- **obs[1]**: `y_position` – 飞行器相对于垫高度的垂直坐标（偏移），reward_usable: true （距离度量）
- **obs[2]**: `x_velocity` – 水平线速度，reward_usable: true （可惩罚/奖励减速）
- **obs[3]**: `y_velocity` – 垂直线速度，reward_usable: true （同上）
- **obs[4]**: `body_angle` – 机体倾角，reward_usable: true （可鼓励保持水平）
- **obs[5]**: `angular_velocity` – 角速度，reward_usable: true （可惩罚过快的旋转）
- **obs[6]**: `left_support_contact` – 左侧支撑脚接触标志（0 或 1），reward_usable: true （可用于判断着陆状态）
- **obs[7]**: `right_support_contact` – 右侧支撑脚接触标志，reward_usable: true （同上）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- **action 0**: `no_engine` – 不点火，飞行器仅受重力、风等影响（节省燃料的动作）
- **action 1**: `left_orientation_engine` – 启动左侧定向引擎，产生力矩调整姿态
- **action 2**: `main_engine` – 启动主引擎，产生向上的推力（可能也有水平分量，取决于姿态）
- **action 3**: `right_orientation_engine` – 启动右侧定向引擎，反向力矩调整姿态

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: 飞行器在目标垫附近稳定停靠（触发 `body_not_awake_or_settled`，且位置靠近原点，速度极小，倾角接近 0）。环境未显式提供成功 flag，需从观测组合推断。
- **failure-like termination**: 
  - `horizontal_position_outside_viewport`：飞行器飞出视口水平边界→失败。
  - `crash_or_body_contact`（但与目标垫安全接触不同的碰撞）：若飞行器身体（非支撑脚）触地或触垫，或姿态严重倾覆后接触→失败。
- **ambiguous termination**: 在远离目标处触发 `body_not_awake_or_settled`（如早期静止在顶部或其他位置）→应视为失败或无效终止。
- **truncation**: 未在 masked step 中出现，可能无时间截断，但实际环境通常有最大步数（未说明，此处只考虑给定的终止条件）。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 没有任何明确可用的字段（step 返回 `{}`）
- **forbidden_or_uncertain_info_fields**: 不允许依赖 info 中的任何字段；如环境中存在的“success”、“failure”等均为未知，不能直接使用。

**推断路径**：通过观测信号间接区分成功/失败。成功着陆特征：`terminated=True` 时 `|x_position| < ε_x`、`|y_position| < ε_y`（接近垫中心）、`|x_velocity|, |y_velocity|` 接近 0、`|body_angle|` 接近 0、`left_support_contact` 与 `right_support_contact` 至少一个为 1（或两者为 1）。失败则表现为出界、远离目标时的静止、或高速/大角度接触。这些均属于 derived_possible 信号。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs`
- `action`
- `next_obs`
- `training_progress`（当前 prompt 未限制，但除非明确需要 curriculum，一般不使用）
- `info` 中没有任何声明可用字段，因此不能依赖 `info`

**禁止使用**：
- `original_reward`（官方奖励被 masked，严禁复原）
- 任何未在上述允许列表中的 info 字段
- 任何未在观测空间中声明的 obs 切片

## 7. 可用于奖励函数的信号
从观测与终止推断的角度：
- **position**:
  - `x_position`（obs[0]）、`y_position`（obs[1]）— 反映到目标垫的欧氏距离或分量距离。
  - 可从 `next_obs` 获取下一步位置。
- **velocity**:
  - `x_velocity`（obs[2]）、`y_velocity`（obs[3]）— 反映接近速度，可用于减速奖励。
- **orientation**:
  - `body_angle`（obs[4]）— 应保持接近 0（水平）。
  - `angular_velocity`（obs[5]）— 抑制过快自旋。
- **contact**:
  - `left_support_contact`（obs[6]）、`right_support_contact`（obs[7]）— 脚部与垫接触，可用于终端着陆检测或稳定性奖励。
- **action/engine**:
  - `action` 本身可用于惩罚引擎使用（燃料惩罚），三个有推力动作为 1,2,3，可赋予不同权重。
- **other**:
  - 终端事件推断：根据终止时的 `next_obs` 状态组合判断成功着陆或失败（出界、坠毁等），提供 derived_possible 奖励/惩罚。

## 8. 不确定或不可用的信号
- `original_reward` — 被显式禁止使用。
- 任何 `info` 字段（如 `info["success"]`, `info["failure"]`）— 不存在于指定 step 中，视为不可用。
- 真实的 `crash` 标签 — 未直接出现在 obs 或 info 中，只能通过位置/速度/角度突变推测（derived）。
- 当前步与下一步的时间间隔或燃料计量 — 未提供。
- 外部风力/重力等环境参数 — 未提供。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with two support legs/feet
  actuator_type: one main thruster (vertical) + two orientation thrusters (left/right)
  contact_structure: left and right foot contacts (binary flags)
primary_objectives:
  - 飞至并稳定停靠在中心目标垫上（位置、速度、姿态收敛）
  - 避免坠毁、出界或非预期的身体硬接触
secondary_objectives:
  - 尽可能节省燃料（减少引擎激活次数或总推力）
  - 快速完成任务（步数短）
main_failure_risks:
  - 过度推力导致超过目标后难以减速
  - 姿态失控使机体以危险角度撞击目标垫或地面
  - 为了避免燃料惩罚而关停引擎时间过长导致坠地
  - 过早减速导致悬停浪费步数，或过晚减速导致软接触失败
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- **role_id**: `progress_towards_target`
  - purpose: 驱动 agent 每步向目标垫靠近并最终到达。
  - why_required: 这是导航类任务的核心，没有此信号 agent 无法学习到达目标区域。
  - usable_signals: `x_position`, `y_position`（当前与下一步），可构造 delta-distance (improvement) 信号。
  - risks: 如果使用纯粹的 proximity（如当前距离的负数），agent 可能在较近但不完美的位置悬停收割 reward，停滞不真正着陆。需使用 improvement 形式（每一步距离减少量），且接近目标时 improvement 自然减小，需配合终端着陆奖励确保最终完成。

- **role_id**: `terminal_landing_event`
  - purpose: 给予成功软着陆的强正奖励，或失败终止的负惩罚。
  - why_required: 没有终端事件，则 improvement 信号最终消失且无法区分成功停靠与半途而废。
  - usable_signals: 终止时 `next_obs` 中的位置、速度、倾角、接触标志（derived_possible），结合终止标志（从环境外部获取，如 `terminated=True`，但 reward 函数仅在有终止时被调用；因此可以在终止时检查 `next_obs` 状态）。
  - risks: 成功条件判断阈值需仔细调优，阈值过严导致成功奖励稀疏，过松可能错误奖励失败的终端状态。

### 10.2 条件职责 conditional_roles
- **role_id**: `orientation_stability`
  - condition_to_use: 当 agent 靠近目标垫（例如距离 < 某个阈值）或在全程保持姿态有一定收益时启用；作为辅助项，避免大角度撞击。
  - usable_signals: `body_angle`, `angular_velocity`。
  - risks: 若权重过大，可能导致 agent 过分关注姿态而不敢移动；应主要用于补偿危险的大角度，而不是苛求精确零度。

- **role_id**: `fuel_efficiency`
  - condition_to_use: 当任务明确要求“使用尽可能少的引擎推力”，可将引擎动作惩罚作为条件信号。最好在训练后期逐渐增加权重（curriculum），以避免早期因保守而卡住探索。
  - usable_signals: `action`（0,1,2,3），可对不同动作赋予不同成本（主引擎成本 > 定向引擎 > 无动作）。
  - risks: 与快速到达目标可能冲突，需权衡；过度惩罚可能导致 agent 选择无推力而迅速坠毁。

- **role_id**: `soft_contact_behavior`
  - condition_to_use: 当靠近垫面且即将接触时，鼓励低速接触；当脚部已经接触且机体质心位于垫上方时，给予小幅奖励。
  - usable_signals: `y_position`（低高度）、`x_velocity`、`y_velocity`、接触标志。
  - risks: 类似姿态稳定，可能影响正常着陆动力学；应仅在 vertical 距离很小时激活。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `dense_survival_bonus`
  - reason: 任务目标是有期限的到达 + 着陆，纯粹的存活奖励会鼓励 agent 原地悬浮而不靠近目标，与“快速到达”冲突。
  - forbidden_or_missing_signals: 没有存活相关的直接信号，且存活不是主目标，故禁用。

- **role_id**: `forward_velocity` 作为主信号
  - reason: 虽然有下坠过程，但主推进引擎方向与机体方向有关，向前速度的定义不明确，且竖直降落为主，更合适的信号是向目标的距离缩小（delta 距离）。
  - 已用 progress_towards_target 替代。

- **role_id**: `reward_shaping_based_on_original_reward`
  - reason: 被任务规则明确禁止。
  - forbidden_signals: original_reward 不可用。

## 11. role_to_signal_mapping

| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| `progress_towards_target` | `x_position`, `y_position` (current & next) | — | `delta_distance = dist(obs[:2]) - dist(next_obs[:2])`，作为主信号 improvement。可配合 `bounded_signal` 裁剪防止异常值。 | 当 agent 到达目标垫后距离不再减少，需终端事件兜底。避免直接用 `proximity` 导致停滞。 |
| `terminal_landing_event` | (derived_possible) `next_obs[0:4]`, `next_obs[6:8]`, termination flag | 显式成功 flag | `sparse_terminal_bon