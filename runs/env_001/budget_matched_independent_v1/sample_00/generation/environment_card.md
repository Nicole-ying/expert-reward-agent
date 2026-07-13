# 匿名环境理解卡片

## 1. 任务目标
这是一个 **2D 飞行器/着陆器轨迹优化** 任务。  
主体从一个视口顶部中心附近出发，带有随机初始力。  
**主目标**：尽快到达视口中央的目标着陆垫，并在其上稳定停靠（速度趋零、姿态稳定、接触垫面）。  
**次目标**：在实现主目标的过程中，尽可能减少发动机推力消耗。  
**不应混淆的目标**：单纯地“存活”或“不坠毁”仅是安全约束，核心是 **精准、快速、节能** 地停靠在指定目标垫上。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: **high**  
reason: 核心目标是到达并停靠在“中央目标垫”这一明确的目标位置，速度、姿态、能耗等均为从属的约束或次优化项，符合“带导航性质的目标到达”任务族。没有多个权重相当的目标冲突。

动力学子类型 dynamics_subtype: **goal_approach_and_soft_contact** （接近目标、减速、稳定接触）

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64（或 float32，字段中有连续值，接触标志为 0/1 的浮点）
- obs[0] (x_position): 水平坐标，相对于目标着陆垫中心。reward_usable: true
- obs[1] (y_position): 垂直坐标，相对于着陆垫高度。reward_usable: true
- obs[2] (x_velocity): 水平线速度。reward_usable: true
- obs[3] (y_velocity): 垂直线速度。reward_usable: true
- obs[4] (body_angle): 机体朝向角度。reward_usable: true
- obs[5] (angular_velocity): 角速度。reward_usable: true
- obs[6] (left_support_contact): 左支撑点是否接触着陆垫（1.0为接触）。reward_usable: true
- obs[7] (right_support_contact): 右支撑点是否接触着陆垫（1.0为接触）。reward_usable: true

**补充说明**：obs 本身是 8 维数组，在每个 step 返回时已经是相对位置，不需要额外转换。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine`，不点火（自然下落/滑行），用于节省燃料或保持状态。
- action 1: `left_orientation_engine`，点燃左侧姿态调整引擎，产生一个使机体逆时针/顺时针转动的力矩（具体方向由环境物理决定，但效果是改变角度）。
- action 2: `main_engine`，点燃主引擎，通常产生向上的推力（对抗重力），并可能附带微小的姿态扰动。
- action 3: `right_orientation_engine`，点燃右侧姿态调整引擎，效果与 action 1 相反，用于反方向调整姿态。

## 5. step 与终止条件分析
### 5.1 终止模式
环境由以下三种条件触发 `terminated = True`：
- **crash_or_body_contact**：主体可能因为速度过高、角度过大或身体部分触地（非着陆垫支撑点）而判定为坠毁，属于失败终止。
- **horizontal_position_outside_viewport**：水平位置超出视口范围，属于失败终止（飞出边界）。
- **body_not_awake_or_settled**：主体“不活跃”或“已稳定”。该条件可能包含两种底层实现：纯粹的物理休眠（长时间无变化）或系统检测到成功着陆稳定。根据任务描述“settled at a central target pad”，**这很可能被设计为成功终止信号**（速度很低、姿态平直、接触着陆垫并稳定）。但也存在未到达目标就休眠的边界情况，需谨慎对待。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** （info 字典为空，没有 `info["success"]` 等字段）
- explicit_failure_flag_available: **false** （同样没有显式标志）
- allowed_info_fields: **无**（info 固定为 `{}`，不能依赖任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用。尤其不能假设存在 `termination_reason`、`success`、`failure` 等。

**解读**：奖励函数必须从观测信号（位置、速度、角度、接触）推断成功或失败，不能直接读取终止原因。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 允许访问 input 参数，但 original_reward 禁止使用
    ...
```
允许使用：
- `obs`：当前 step 的观测（8 维数组）
- `action`：刚刚执行的动作（整数 0~3）
- `next_obs`：执行动作后的下一观测（8 维数组）
- `info`：当前 step 的 info 字典，但本例中恒为 `{}`，可忽略
- `training_progress`：仅在 prompt 明确允许训练阶段调度时才使用，默认可不用

禁止使用：
- `original_reward`：**完全不允许访问、拷贝或推导**
- 任何未在合法参数中声明的信号（如环境内部状态、官方奖励计算逻辑）
- 未声明的 info 字段
- 对 obs 的切片如含义不明，需以本文档的字段含义为准

## 7. 可用于奖励函数的信号
以下信号均来自 `obs` 或 `next_obs`，可直接用于构造奖励。
- **位置相关**：
  - `x_position`（obs[0]）、`y_position`（obs[1]），相对的水平和垂直位移，可直接用于算距离（如欧几里得距离）。
- **速度相关**：
  - `x_velocity`（obs[2]）、`y_velocity`（obs[3]），可用于惩罚过大速度或鼓励减速。
- **姿态相关**：
  - `body_angle`（obs[4]），可用于惩罚偏离竖直（0°）的姿态。
  - `angular_velocity`（obs[5]），可用于鼓励稳定性。
- **接触相关**：
  - `left_support_contact`（obs[6]）、`right_support_contact`（obs[7]），双侧接触（均为 1.0）可视为成功着陆在垫上，可用于触发接触奖励或判断稳定。
- **动作/能耗相关**：
  - `action`：可判断是否使用主引擎或姿态引擎，用于惩罚能量消耗。

## 8. 不确定或不可用的信号
- **目标是否已到达的显式标志**：无（`info` 为空，终止条件被湮灭，不能直接读取“成功着陆”）。
- **燃料量/能量消耗绝对值**：观测中无此字段，只能通过动作频率间接估计。
- **距离水平/垂直方向分解的阶段性信号**：需人工从 `obs` 位置分量计算。
- **外部干扰力（如风）**：观测中未提供，无法直接感知。
- **时间步计数**：环境未显式传递，但可利用外部 `training_progress` 进行隐式推断（不推荐作为主要奖励依据）。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body (vehicle-like)
  actuator_type: discrete thrusters (main engine + two orientation engines)
  contact_structure: two support contact points (left/right) that must touch the landing pad simultaneously for successful landing
primary_objectives:
  - Reach the target pad (minimize horizontal and vertical distance to origin)
  - Make safe and stable contact (both contact flags = 1, low velocities, near-zero angle)
secondary_objectives:
  - Minimize fuel/energy usage (reduce main engine and unnecessary orientation firings)
  - Reach the pad quickly (implicitly encouraged by dense goal-proximity reward)
main_failure_risks:
  - Crashing due to excessive touchdown speed or wrong angle
  - Drifting out of horizontal viewport
  - Settling in place far from the pad (wasting steps or stuck)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `goal_proximity_reward`
  - purpose: 驱动主体向目标（原点）移动，减少到原点的欧氏距离或分量距离。
  - why_required: 导航到达的核心，否则主体可能漫无目的。
  - usable_signals: `x_position`, `y_position` （从 `obs` 或 `next_obs` 提取）
  - risks: 若仅用距离，可能导致高速冲过目标；需配合其他职责。

- **role_id**: `safe_landing_reward`（成功接触＋稳定）
  - purpose: 在最终稳定在着陆垫上时给予一次性正向奖励，或检测双侧接触并低速低角速度时持续奖励。
  - why_required: 区别于“经过目标”，必须明确鼓励停靠动作。
  - usable_signals: `left_support_contact`, `right_support_contact`, `x_velocity`, `y_velocity`, `body_angle`, `angular_velocity`（可选）
  - risks: 若过度依赖接触信号，可能诱导提前猛烈撞击垫面获取接触；需结合低速和低角速度条件。

- **role_id**: `energy_penalty`
  - purpose: 对使用发动机的动作施加惩罚，鼓励无动作滑翔期。
  - why_required: 次目标“尽可能少使用发动机推力”。
  - usable_signals: `action` （0=无动作不罚，1/2/3×相应权重）
  - risks: 若惩罚过重，可能导致主体不敢动作，难以调整位置。

### 10.2 条件职责 conditional_roles
- **role_id**: `orientation_penalty`
  - condition_to_use: 全程或仅在接近目标时（距离较小时）启用。全程启用有助于防止翻转，接近时启用则确保竖直着陆。
  - usable_signals: `body_angle` （obs[4]）
  - risks: 过早强惩罚可能干扰初始姿态调整的自由度，可根据距离动态缩放。

- **role_id**: `velocity_smoothing_penalty`
  - condition_to_use: 在接近目标（距离小于阈值）或已经接触垫面时，对速度施加更严厉的惩罚。
  - usable_signals: `x_velocity`, `y_velocity`，结合位置距离决定强度。
  - risks: 若阈值不当，可能妨碍正常减速过程；可使用连续缩放。

- **role_id**: `angular_damping_penalty`
  - condition_to_use: 当接触垫面或即将着陆时，惩罚角速度。
  - usable_signals: `angular_velocity` （obs[5]）
  - risks: 全程应用可能降低姿态调整的敏捷性。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `explicit_success_bonus`
  - reason: 环境中没有 `info["success"]`，无法得知真实成功判断；依赖伪造的成功信号会误导训练。
  - forbidden_or_missing_signals: 无 `success` 标记。

- **role_id**: `completion_time_penalty`（按时间步惩罚）
  - reason: 虽然任务说“尽可能快”，但没有提供步数计数信息，且这种全局时间惩罚可能被误用，不如用距离减少的效率来隐式鼓励快速接近。当前环境无 `time_step` 信号，不建议作为独立职责。
  - forbidden_or_missing_signals: 每步时间指标不可得。

- **role_id**: `external_disturbance_compensation`
  - reason: 观测没有风等外部力的信息，无法补偿。
  - forbidden_or_missing_signals: 无。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity_reward | x_position, y_position | — | dense_state_signal, bounded_signal (如 -distance) | 常用奖励变化量 Δ(-distance) 或当前 -distance |
| safe_landing_reward | left_contact, right_contact, velocities, body_angle, angular_velocity | 官方success标志 | condition_on_contact_and_stability | 可设计为当 contact=2 且 |vel|、|angle| 小于阈值时给予较大正奖励 |
| energy_penalty | action (0~3) | 能量绝对值 | discrete_action_penalty | 仅对 1,2,3 动作施加固定或比例惩罚，0 无罚 |
| orientation_penalty | body_angle | — | quadratic_penalty, distance_to_target_angle (0) | 可与距离缩放，减少远离目标时的干扰 |
| velocity_smoothing_penalty | x_velocity, y_velocity, distance_to_goal | — | velocity_norm_penalty × distance_factor | 越近惩罚越重 |
| angular_damping_penalty | angular_velocity | — | absolute_penalty | 着陆地标触发 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 主体悬停在高处不动或反复上下摆动 | y_pos 几乎不减小，距离停滞；动作可能是频繁小推力维持高度。 | 调整 `goal_proximity_reward` 权重，或增加对高度下降的线性奖励。 |
| 主体高速撞击着陆垫导致坠毁 | 终止时 crash_or_body_contact 且 contact 可能短暂出现但速度极大。 | 强化 `safe_landing_reward` 中的速度惩罚，或增加接近目标时的减速奖励。 |
| 主体超出水平视口 | x_pos 绝对值持续增大，最终终止。 | 增加对横向偏移的惩罚，或让 `goal_proximity_reward` 包含水平分量的权重提升。 |
| 主体稳定着陆在目标外（如垫子边缘外的地面），但触发了休眠终止 | 终止时 contact 可能为0或仅有单侧接触，而速度和角度很小。 | 确保 `safe_landing_reward` 严格要求双侧接触，否则不能获得高奖励。 |
| 过度使用姿态引擎导致燃料浪费但未进步 | 动作 1 或 3 频繁，角度波动大，前进不大。 | 增加 `orientation_penalty` 强度，或对非必要姿态调整的动作施加更高 `energy_penalty`。 |

---