# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境事实、任务画像、奖励职责拆解、职责-信号映射；
2. expert_reward_context.md：固定专家 Schema，包括任务类型示例和 Formula Operator Library；
3. optional masked_step_source：默认不提供，除非调试开启。

你的任务不是机械选择某个 skeleton，而是：
1. 读取 environment_card.md 的 `expert_task_profile`；
2. 读取 `reward_role_decomposition`，明确 mandatory / conditional / avoid roles；
3. 使用 `role_to_signal_mapping` 检查每个职责可用的 obs/action/info 信号；
4. 从 expert_reward_context.md 的 Formula Operator Library 中为每个 selected role 选择数学形式；
5. 生成第一版奖励函数 `reward_v1.py`，并附带简短设计说明。

# Expert Schema 使用规则

- environment_card.md 中的 `reward_role_decomposition` 优先级高于 expert_reward_context.md 的模板。
- expert_reward_context.md 只提供专家模板和公式算子，不是固定答案。
- 先选 role，再选 signal，再选 formula operator，最后才写代码。
- 如果某个 role 没有可用信号，必须放入 excluded_roles，不得硬写。
- 如果 task_profile 与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不允许因为模板里提到某个 role 就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康/安全约束；效率、能耗、复杂门控和动态权重默认后续迭代再加入。

# 总体设计原则

- 从简单到复杂，但”简单”不等于只有一个组件。
- 不要用”最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 写完 reward 后自检：① 每个终止条件是否有前兆软信号？② 任务目标是否有直接的进度信号？③ 动作维度 ≥ 6 时，是否缺少效率约束（即使权重很小）？
- 不要机械照抄 expert template 或 formula operator。
- 不要使用 original_reward。
- 不要计算 fitness_score 或 fitness_score components。
- 不要使用未声明的 info 字段，例如 info["success"]、info.get("success")。
- 不要使用未声明的 obs 切片，例如 obs[0:3]。
- 只能使用 environment_card.md 声明的观测维度和索引，不得自行扩展为未声明的二维、三维或其他结构。
- 如果 explicit_success_flag_available=false，不要把 terminal_success_reward 写成 v1 核心项。
- 如果 explicit_failure_flag_available=false，不要把 terminal_failure_penalty 写成 v1 核心项。
- 允许使用 obs 和 next_obs 的逐 index 变量。
- 尽量让奖励平滑；需要距离、速度等连续项时，优先使用连续函数。
- 如果需要 sqrt，禁止 import numpy，使用 `** 0.5`。
- 如果想使用 exp 形式的平滑变换，禁止 import numpy；可以使用 `2.718281828 ** (...)`，并显式写 temperature 参数。

# 任务无关设计原则

## 原则 1：信号可用性优先

- 先检查 environment_card.md 中声明的可用信号、禁止信号和 role_to_signal_mapping。
- 只有当信号确实存在于环境接口中时，才设计依赖该信号的组件。
- 如果 explicit_success_flag_available=false，不要使用 terminal_success_reward。
- 如果 explicit_failure_flag_available=false，不要使用 terminal_failure_penalty。
- 不要发明未声明的 info 字段或 obs 切片。

## 原则 2：稠密性

- 优先选择每步都能提供有意义梯度的连续信号。
- 二值条件信号触发率过低时等于摆设。
- 连续函数、bounded 函数、soft proxy 通常比硬阈值更利于学习。

## 原则 3：尺度与平衡

- 不同组件的量级应大致可比，不要让一个组件在数值上统治其他组件。
- 约束/惩罚不应无条件压制任务驱动力；具体尺度必须结合触发频率、数学形态和预期行为判断。
- 差分信号、持续状态奖励和稀疏事件奖励具有不同时间语义，不能仅凭步均值比例判断谁更重要。

## 原则 4：信号冲突

- 不要同时大权重使用两个计算同一物理量的信号。
- 不要让惩罚项压制探索；过严姿态/速度/动作约束可能导致 agent 不敢行动。
- soft_health_gate 比强全局惩罚更适合处理“前进但失稳”的早期问题。

## 原则 5：阶段条件

- v1 阶段避免过早引入效率/动作代价；agent 应先学会任务方向，再优化效率。
- 复杂门控、动态课程、强能耗项默认后续迭代再加入。
- curriculum_weighting 只有当 training_progress 明确允许且任务确有阶段性冲突时才使用。

## 原则 6：可利用风险

- 每个组件都要考虑 agent 可能找到的捷径。
- 只奖励速度可能导致 velocity_burst_then_fall。
- 只奖励存活可能导致 stand_still 或 hover。
- 只奖励接触可能诱导 contact reward hacking。
- 直接奖励 vertical activity 可能诱导原地弹跳。

# role-based component budget

v1 推荐使用 2~4 个组件，按以下角色组织。专家模板和公式算子只提供设计启发，不限制你组合、变形或创造适合当前环境的新信号。

## 必须包含

**1 个主学习信号。** 这是 reward 的核心驱动力，告诉 agent “做什么能得分”。主信号的特征：
- 每步都有梯度；
- 与任务目标直接相关；
- 在策略学习中承担主要任务驱动作用；
- components key 应准确描述其物理或任务含义，不强制命名为 `progress_reward`。

## 允许包含（按需，不是必须全加）

- **0~2 个稳定/安全/健康约束。** 如果任务需要控制速度、姿态、身体高度、角速度等，可以加入轻量惩罚或 soft gate。约束的角色是“方向盘”而非“刹车”。
- **0~1 个任务完成近似信号。** 如果环境没有显式 success flag 但需要在 agent 接近完成时给予额外引导，可以用多条件组合的 soft proxy。proxy 必须由多个连续条件组合，不能直接伪造 success flag。
- **0~1 个效率/动作代价。** v1 默认不加或极小权重；能耗优化通常留到后续迭代。

## 默认不在 v1 使用

- terminal_success_reward（需显式 success flag，且 flag 在 info 中实际可用）
- terminal_failure_penalty（需显式 failure flag 或明确 termination_reason）
- 强 gated_reward（多阶段门控，复杂且容易过严）
- dynamic_curriculum_reward（依赖训练进度，v1 无历史参考）
- action_smoothness_penalty（如果没有 previous action/history，不得使用）

# 输出格式要求

函数签名必须完全一致：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

最终 reward 函数输出必须包含：
1. total_reward: float
2. components: dict，记录 individual reward components

首选返回格式：
```python
return float(total_reward), components
```

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
- 不要写 import。
- 不要写 class。
- 不要写 try/except。
- 不要写 eval/exec/open。
- 不要创建额外函数。
- 不要引入新的输入变量。
- 不要传 self；当前项目接口不是 Eureka 原版 self 接口。
- 不要使用 self attributes。
- 不要使用原始环境 reward。
- components 必须是 dict。
- components 只包含被加到 total_reward 的组件（A、B、C），不包含 total_reward 本身。

# Markdown 输出要求

输出必须是 Markdown，但第一个 Python code block 必须只包含完整且可执行的 `compute_reward` 函数，因为 parser 会抽取第一个 Python code block。

格式：

# reward_v1.py

```python
def compute_reward(...):
    ...
```

# reward_v1 设计说明

必须简要说明：
- selected task_family / dynamics_subtype；
- selected reward roles；
- role_to_signal_mapping；
- 每个 role 选择的 formula operator；
- excluded roles 及原因；
- 为什么没有使用 terminal_success_reward / terminal_failure_penalty；
- 哪些职责留到后续迭代；
- 训练后应该观察哪些 failure modes。

```

## User Prompt

```markdown
# environment_card.md

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



# expert_reward_context.md

# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

不要反过来先套某个 skeleton 名称。模板只提供专家思考方式，不构成封闭候选集合。

---

## 1. Expert Schema 使用规则

- environment_card.md 中的 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

每个算子包含：数学形式、适用场景、触发证据、反模式。

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2` 或 `w * exp_form`
    凸化形式在 signal 较大时提供更强梯度。触发证据：episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小——说明 agent 满足于低水平稳态，需要凸化奖励来打破。
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
    hinge 只在超出安全区间时生效，避免在安全范围内持续惩罚正常波动。触发证据：约束组件的 active_rate≈100% 但 terminated 率仍然很高——说明"全时惩罚"没有给 agent 安全探索空间，它无论怎么调整都被罚。
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：线性正奖励可能导致慢速平台；凸化形式若权重过大可能诱导极端行为；hinge 的 threshold 设太宽则防护不足。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - 平滑压缩: `x / (1 + abs(x))`
  - 倒数衰减: `1 / (1 + k * abs(error))`
  - 线性衰减: `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。
- 触发证据：某个信号的 episode_sum_mean 出现极端值（远大于其他组件），说明无界形式被 exploit。
- 风险：threshold 过小会导致反馈饱和或无梯度。
- 反模式：不要用 bounded_signal 替代 hinge penalty——如果目标是"只在越界时惩罚"，用 dense_state_signal 的 hinge 形式，不要用 bounded 包围。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 触发证据：有明确的目标度量（如到目标的距离）且该度量在 episode 中单调递减时 agent 表现好。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。
- 反模式：不要对速度类信号用 improvement_delta——持续速度本身已经是"进步"，delta 会退化为噪声。

### 2.4 potential_based_shaping
- 适用职责：有明确 potential function 的任务塑形。
- 常见形式：`gamma * Phi(next_obs) - Phi(obs)`
- 使用条件：能够从环境信号定义合理的 Phi。
- 风险：错误 Phi 会误导策略；reward_v1 不默认使用，除非任务天然适合。

### 2.5 quadratic_penalty
- 适用职责：姿态误差、角速度、动作幅度、速度等轻量约束。
- 常见形式：`-w * error**2` 或 `-w * sum(action_i**2)`
- 使用条件：约束信号可观测，且不应压制主学习信号。
- 风险：权重过大会导致 agent_afraid_to_move 或 over_conservative_policy。
- 触发证据：某维度出现高频大幅波动或极端值，但没有触发终止——说明需要轻量抑制而非硬约束。
- 反模式：不要对"有明确安全边界"的信号用 quadratic_penalty（如身体高度必须在 0.2-1.0）。quadratic 从中心开始罚，会让 agent 困在中心不敢动；应改用 hinge 形式只在边界附近生效。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * gate_factor`，gate_factor 在身体状态恶化时从 1 平滑衰减到 0。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (signal - danger) / margin))`
- 使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
- 触发证据（关键）：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍然 >0——说明 agent 在"先冲后死"，需要 gate 在健康恶化时切断主奖励，而不是加一个独立惩罚。
- 风险：gate 太严格会抑制探索；gate 的衰减区间应设在"接近危险但尚未终止"的范围内。
- 反模式：不要用"加大独立惩罚系数"替代 gate。如果 terminated 是因为身体状态越界，单纯加大该状态的惩罚（Level 1）通常不如将其作为 gate 乘到主奖励上（Level 2），因为惩罚只在越界后才生效，gate 在越界前就开始衰减主信号。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 触发证据：agent 能在各个子条件上分别取得进展，但无法同时满足——说明缺一个"联合满足"的引导信号。
- 风险：乘积容易塌缩（一个 factor 趋近 0 则整体为 0）；使用 `(factor_1 + factor_2 + ...) / n` 或几何平均 `(factor_1 * factor_2 * ...) ** (1/n)` 可缓解。
- 反模式：不要用二值条件做乘积——每个 factor 必须是连续函数，否则乘积退化为稀疏信号。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

---

## 3. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"直接定位到"该选哪个算子做 Level 2 变换"。
不要求组件名完全匹配；以数学语义和训练表现证据为准。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |


```
