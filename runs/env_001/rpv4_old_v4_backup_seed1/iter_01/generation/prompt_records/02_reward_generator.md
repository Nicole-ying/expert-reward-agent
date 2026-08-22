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

- 从简单到复杂，但“简单”不等于只有一个组件。
- 不要用“最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
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
```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code

**2. components: dict**，记录 individual reward components。

首选返回格式：
```python
return float(total_reward), components
```

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
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
该任务要求控制一个具有两条支撑腿的 2D 刚体飞行器，从视口顶部中央附近开始（带有随机初始力），**快速、低燃料地到达并稳定停靠在中央目标垫上**。核心是到达指定目标位置并实现安全软着陆，附属目标是最小化发动机使用次数（燃料）与总时间。代理人必须学会平滑接近目标、减速、保持竖直姿态并产生双腿的稳定接触。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: **high**  

reason: 任务的唯一核心是到达明确的空间目标（中央目标垫），其他如速度、姿态、燃料等都属于保障性约束或效率优化，不存在权重相当的冲突目标。因此，粗粒度任务族属于导航目标到达类。进一步，由于要求“减速、保持稳定方向、安全接触”，其动力学表现为接近目标后的柔和减速与接触，故动力学子类型为 **goal_approach_and_soft_contact**。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64
- obs[0]: `x_position` — 飞行器相对于目标垫中心的水平坐标。可用于奖励对靶心的逼近。reward_usable: true
- obs[1]: `y_position` — 飞行器相对于目标垫表面的垂直坐标。可用于奖励下降与接近。reward_usable: true
- obs[2]: `x_velocity` — 水平线速度。可用于惩罚大水平漂移。reward_usable: true
- obs[3]: `y_velocity` — 垂直线速度。控制软着陆冲击，极其重要。reward_usable: true
- obs[4]: `body_angle` — 刚体倾角。竖直向上时接近 0。奖励维持竖直。reward_usable: true
- obs[5]: `angular_velocity` — 角速度。鼓励姿态稳定。reward_usable: true
- obs[6]: `left_support_contact` — 左腿接触标志（0/1）。表示左腿已与地面接触。reward_usable: true
- obs[7]: `right_support_contact` — 右腿接触标志（0/1）。表示右腿已与地面接触。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` — 不开启任何引擎，仅受重力/惯性影响。
- action 1: `left_orientation_engine` — 启动左侧姿态引擎，主要产生逆时针力矩（可能附带微小侧向力），用于校正右倾。
- action 2: `main_engine` — 启动主引擎，产生向上的推力，是主要的减速和高度控制手段。
- action 3: `right_orientation_engine` — 启动右侧姿态引擎，主要产生顺时针力矩，校正左倾。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 当 agent 以低速度、接近竖直的姿态通过双腿接触目标垫，且不再运动（`body_not_awake_or_settled`）时触发。这通常意味着安全着陆成功。
- **failure-like termination**:
  - `crash_or_body_contact`：飞行器任意部位（特别是非腿部）与地面或其它物体发生硬碰撞，导致坠毁。
  - `horizontal_position_outside_viewport`：飞行器水平漂出窗口，视为失控。
- **ambiguous termination**: 若仅因 `body_not_awake_or_settled` 终止，但未满足成功条件（例如悬浮在空中静止，或偏离目标垫静止），则不属于成功；但这类情形在合理动力学下罕见。
- **truncation**: 任务描述中未出现时间截断条件，但实际训练中可能存在未声明的最大步数截断；当前 step 源无 `truncated` 信息，奖励函数中不应依赖该假设。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
  step 返回的 `info` 固定为 `{}`，没有任何成功/失败标志或终止原因字段。奖励函数必须完全从 `next_obs` 的物理状态（位置、速度、接触）中推断是否成功着陆或发生硬碰撞。
- **allowed_info_fields**: 无（info 为空）。
- **forbidden_or_uncertain_info_fields**: `info["success"]`, `info["failure"]`, `info["termination_reason"]` 等未声明字段均禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用的参数：
- `obs`：合法，上一时间步的观测。
- `action`：合法，刚刚执行的离散动作。
- `next_obs`：合法，动作执行后的下一时刻观测，是奖励的核心依据。
- `info`：仅 `{}`，无可用字段，禁止从中提取任何信息。
- `training_progress`：若 prompt 未明确允许，默认禁止使用。

禁止使用的参数：
- `original_reward`：已掩盖，严禁引用任何形式的官方奖励。
- 任何未在观察空间中声明的 obs 切片。
- 任何未在 info 中声明的字段。

## 7. 可用于奖励函数的信号
- **position**: `next_obs[0]`（水平差距）, `next_obs[1]`（垂直高度）。两者可直接用于构建距离/接近奖励。
- **velocity**: `next_obs[2]`, `next_obs[3]`。尤其垂向速度对软着陆至关重要。
- **orientation**: `next_obs[4]`（倾角）, `next_obs[5]`（角速度）。可用于姿态稳定性奖励。
- **contact**: `next_obs[6]`, `next_obs[7]`。表示双腿是否触地，是判断着陆成功的关键二值特征。
- **action/engine**: 可利用 `action` 实现燃料惩罚（例如对 1,2,3 动作添加负奖励）。
- **other**: 无其他活跃信号。

## 8. 不确定或不可用的信号
- **显式成功/失败标志**：不存在。
- **步数或时间**：未观察到任何时间信号，不能直接构造“尽快完成”奖励，但可通过每步微小负奖励模拟时间惩罚，需谨慎使用。
- **腿部力量或接触力**：仅有 0/1 接触指示，无连续力/压强信息。
- **视口外惩罚细节**：不清楚水平超界时的具体位置值（可能被截断）, 奖励设计时不依赖该事件自身。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body with two support legs
  actuator_type: single main vertical thruster + two lateral orientation thrusters
  contact_structure: two point contacts, each binary collision flag
primary_objectives:
  - reach and rest at center target pad (zero relative horizontal position, zero altitude over pad)
  - achieve stable, upright landing at near-zero linear and angular velocities
secondary_objectives:
  - minimize main and orientation engine usage (fuel efficiency)
  - implicitly minimize episode length (fast settling), though no explicit timer
main_failure_risks:
  - hard crash from excessive vertical speed or body contact other than legs
  - drifting out of horizontal bounds
  - landing upside down or with excessive tilt, causing crash
  - failure to reach target pad or remaining airborne until possible truncation
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `goal_proximity`  
  purpose: 引导 agent 持续缩短与目标垫的水平距离和垂直距离。  
  why_required: 核心导航目标，必须始终施加。  
  usable_signals: `next_obs[0]`, `next_obs[1]`  
  risks: 若权重大于安全信号，可能导致直接高速坠落接触。

- **role_id**: `soft_landing_velocity_penalty`  
  purpose: 在 agent 接近垫面时，严厉惩罚垂向和水平速度，促使减速。  
  why_required: 实现“软接触”和稳定停靠，是避免坠毁的关键。  
  usable_signals: `next_obs[2]`, `next_obs[3]`  
  risks: 可能造成悬停或延长停留时间；需与 `goal_proximity` 平衡。

- **role_id**: `upright_orientation`  
  purpose: 维持身体倾角接近 0，并抑制角速度。  
  why_required: 倾斜过大会导致单腿先触地、另一腿悬空或身体侧面碰撞地面，直接触发 `crash_or_body_contact`。  
  usable_signals: `next_obs[4]`, `next_obs[5]`  
  risks: 过强的角度约束可能妨碍必要的姿态调整。

- **role_id**: `contact_success`  
  purpose: 当双腿同时接触垫面（`left_contact=1` and `right_contact=1`）时，给予大规模正奖励，标记着陆成功。  
  why_required: 这是唯一明确的“着陆完成”信号；结合速度几乎为零时，即可确信安全着陆。  
  usable_signals: `next_obs[6]`, `next_obs[7]`  
  risks: 可能诱导 agent 在目标垫外强行伸出双腿触地，因此必须与位置奖励联合使用。

### 10.2 条件职责 conditional_roles
- **role_id**: `fuel_efficiency`  
  condition_to_use: 当训练进入中后期，基本着陆能力已稳定时逐步引入。  
  usable_signals: `action` (对 1,2,3 给予小负奖励)  
  risks: 早期引入会严重限制探索，导致 agent 不敢使用主引擎。

- **role_id**: `angular_velocity_damping_near_surface`  
  condition_to_use: 当 `next_obs[1]`（高度）低于某个阈值时增强，防止接地瞬间旋转。  
  usable_signals: `next_obs[5]`  
  risks: 若阈值过高，可能干扰下降过程中的姿态校准。

- **role_id**: `per_step_time_penalty`  
  condition_to_use: 可选，用于鼓励快速完成；但可能引发冒险行为，仅在基本安全着陆率高时谨慎加微负值。  
  usable_signals: 可直接返回常数负值，不依赖任何观测。  
  risks: 过早使用会导致 agent 优先速度而非安全。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `explicit_success_flag_reward`  
  reason: 因为 info 为空，没有可用的显式成功标志。  
  forbidden_or_missing_signals: `info["success"]` 等所有 info 字段。

- **role_id**: `terrain_awareness` 或 `force_based_penalty`  
  reason: 没有连续接触力测量，仅二值接触，无法构建精细接触力惩罚。  
  forbidden_or_missing_signals: 连续力/压强数据。

- **role_id**: `trajectory_style`（例如最小作用量或平滑性）  
  reason: 当前高维观测已足够，且离散动作下平滑性很难有效量化，暂不引入额外复杂度。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | `next_obs[0]`, `next_obs[1]` | – | `euclidean_distance`, `bounded_signal` | 可直接使用水平、垂直偏差的平方和或绝对值。 |
| soft_landing_velocity_penalty | `next_obs[2]`, `next_obs[3]` | – | `quadratic_penalty`, `dense_state_signal` | 主用垂向速度（`next_obs[3]`），水平速度次之。 |
| upright_orientation | `next_obs[4]`, `next_obs[5]` | – | `absolute_penalty`, `quadratic_penalty` | 角度和角速度可独立惩罚。 |
| contact_success | `next_obs[6]`, `next_obs[7]` | – | `logical_and` + `large_bonus` | 仅当两腿同时触地时发放高分。 |
| fuel_efficiency (conditional) | `action` (0,1,2,3) | – | `sparse_action_penalty` | 对 1,2,3 动作给予小额负奖励。 |
| angular_velocity_damping_near_surface (conditional) | `next_obs[5]`, `next_obs[1]` (for gating) | – | `gated_quadratic` | 以高度阈值激活。 |
| per_step_time_penalty (conditional) | 无（常数） | – | `constant_negative` | 极小的常数负值，需后期启用。 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高速撞击目标垫 | `next_obs[3]` 绝对值很大，终止时双腿虽接触但奖励可能过早高分。 | 增加垂向速度惩罚权重，或缩小 `contact_success` 激活时的速度阈值，仅在低速接触时给予高分。 |
| 悬停不降 | 高度长期不变，速度接近零，但未接触垫子；若无时间截断则不会终止。 | 引入微小的高度-时间惩罚或势能函数，或在 `goal_proximity` 中增加垂向迫近力；也可添加逐步减弱悬停奖励的机制。 |
| 过度倾斜着陆 | `body_angle` 远离 0，最终只有单腿接触，可能触发 `crash_or_body_contact`。 | 加强 `upright_orientation` 权重，并在接近地面时增加角度惩罚。 |
| 在目标垫外触地 | 双腿接触但 `x_position` 偏离 0 较大。 | 在 `contact_success` 发放时乘以与 `x_position` 相关的高斯衰减项，或在接触时仍保留强烈的水平位置惩罚。 |
| 过度使用引擎 | 连续启用主引擎或姿态引擎，尽管已稳定着陆或悬停。 | 逐步引入 `fuel_efficiency` 惩罚，并确保着陆后 agent 学会执行 `no_engine`。 |
| 学习到“牺牲一腿”策略 | 始终保持大倾角，仅让一条腿接触，可能仍能触发 `body_not_awake_or_settled` 而误判为成功？ | `contact_success` 必须要求双腿同时接触，否则不给予成功奖励；同时加强倾角惩罚。 |



# expert_reward_context.md

# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

---

## 1. Expert Schema 使用规则

- environment_card.md 中的任务画像和可用信号优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role（任务需要什么类型的奖励信号），再选 signal（哪个观测维度承载这个 role），再选 formula operator（用什么数学形式表达），最后写代码。
- 如果某个 role 需要的信号在观测空间中不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- reward_v1 以主学习信号和必要的稳定/安全约束为重点。效率、能耗、复杂门控和动态权重可以在后续迭代中按需加入，但不应因"模板没列"而排除合理的设计。

---

## 2. 信号完备性自查清单

在完成初始设计后，逐一检查以下信号类型是否被覆盖——不是每个任务都需要全部，但每一项的缺失应是有意选择：

- **主进展信号**：agent 朝任务目标前进时是否获得正向反馈？该信号是否每步都有梯度？
- **灾难性失败信号**：是否存在明确的终止惩罚（如摔倒、飞出边界）？如果观测中可推断失败状态，是否给予了足够强的负向信号？
- **效率/代价信号**：连续动作空间中是否有能量消耗或控制代价约束？离散动作空间中是否有不必要的动作惩罚？
- **任务完成信号**：终止条件中是否包含 success-like 条件？相应的观测是否可被用来构造任务完成的软近似信号？
- **健康/稳定约束**：agent 是否因缺少姿态/速度/位置约束而产生不安全行为？

---

## 3. Formula Operator Library

每个算子包含：数学形式、使用条件、适用证据。

### 3.1 dense_state_signal
数学形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2`
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
使用条件：该状态信号每步可观测，且与某项任务职责直接相关。
适用证据：
  - 凸化 → episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小（agent 满足于低水平稳态）。
  - hinge → 约束组件的 active_rate≈100%（全时惩罚）但 terminated 率仍高，说明 agent 在安全范围内也被持续惩罚，需要只在越界时生效的 hinge。
风险：线性正奖励在信号平台期无梯度；凸化权重过大可能诱导极端行为；hinge 的 threshold 需根据环境卡片的观测范围设定。

### 3.2 improvement_delta
数学形式：`old_measure - new_measure`（期望减少时）或 `next_value - current_value`（期望增加时）
使用条件：obs 和 next_obs 中存在可比较的标量度量，该度量沿最优路径应单调变化。
适用证据：有明确的进展度量（位置、距离、高度、角度等），且该度量的变化比瞬时速率更能反映真实进展。
与 dense_state_signal 的选择：如果要鼓励"处于某种好状态"，用 `w * signal`。如果要鼓励"朝好方向改变"，用 delta。delta 的优势是 agent 无法在好状态上停滞不前，必须持续改善。适合：agent 当前的绝对状态值不能完全反映进展（如位置——站在原点不动 vs. 走到终点但位置绝对值可能相同）。
注意：对观测中直接给出的速度信号（如 `horizontal_velocity`）不要做 delta——速度本身已经是变化率。对观测中的位置/角度/距离类信号优先考虑 delta。

### 3.3 potential_based_shaping
数学形式：`potential(next_obs) - potential(obs)`
使用条件：(1) 任务有一个可量化的进展度量（如位置、距离、高度）；(2) 该度量沿最优路径应单调变化；(3) 能从观测中构造一个标量的 potential function。
如何构造 potential：从观测中选择一个在任务完成时达到极值、且沿最优路径单调变化的信号（或信号组合）。potential 的计算只能依赖观测，不能依赖环境内部状态。
与 improvement_delta 的关系：两者数学上等价。potential_based_shaping 的优势在于允许将多个信号编码到一个 potential 中（如同时考虑位置和姿态），而 improvement_delta 通常用于单个度量。
风险：potential 若与任务目标不一致会系统性地误导策略。reward_v1 中如果存在天然的进展度量，优先使用 improvement_delta 的简单形式；当需要组合多个信号构造进展度量时，使用 potential_based_shaping。

### 3.4 quadratic_penalty
数学形式：`-w * error**2` 或 `-w * sum(action_i**2)`
使用条件：约束信号连续可观测，惩罚不应压制主学习信号。用于轻量抑制——需要约束但不至于触发终止的行为。
适用证据：某维度出现高频大幅波动或极端值但未触发终止。
与 hinge 的选择：如果约束有明确的安全边界（如身体倾角超过 X 度必摔），用 hinge（3.1）。如果只是希望"越小越好"没有硬边界（如控制代价、小幅抖动），用 quadratic。
风险：权重过大导致 agent 不敢行动。

### 3.5 soft_health_gate
数学形式：`main_reward * gate_factor`，gate_factor ∈ [0, 1] 在身体状态恶化时平滑衰减。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (safe_bound - current) / margin))`
使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
适用证据：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍 >0——agent 在"先冲后死"，需要在健康恶化时切断主奖励而非额外加罚。
风险：gate 太严格抑制探索；衰减区间应设在"接近危险但尚未终止"的范围内。

### 3.6 terminal_event
数学形式：`if failure_condition: reward = -PENALTY`（硬覆盖 per-step 奖励），或 `if success_condition: reward = +BONUS`
使用条件：(1) 存在可从观测推断的灾难性失败状态（如身体倾角超过阈值 + 接触地面）或任务完成状态；(2) 环境 info 为空因此无法直接读取终止原因。
如何构造：不要依赖 info 字段判断终止原因。可从观测推断：摔倒 → hull_angle 突然偏转 + 身体位置急剧下降；到达终点 → 持续前进中 episode 突然终止（truncated）；出界 → 位置坐标超出有效范围。
适用证据：agent 频繁触发某种终止模式，但当前奖励没有针对该模式提供差异化信号——比如所有终止回合 reward 都一样，agent 无法区分成功和失败。
与 hinge/gate 的区别：hinge 在越界前提供连续梯度，gate 在恶化时衰减主信号。terminal_event 在事件发生的那一刻提供硬信号——没有梯度，但语义明确（"这就是你应该避免/追求的结果"）。

### 3.7 action_efficiency
数学形式：`-w * sum(|action_i|)` 或 `-w * sum(action_i**2)`
使用条件：动作空间 ≥ 2 维连续控制，且任务包含隐含的效率需求（如 locomotion、manipulation）。
适用证据：agent 学会完成任务但动作幅度异常大、能耗高——说明缺效率约束。通常系数较小（主信号 per-step 的 1-5%），避免压制探索。
注意：离散动作空间通常不需要此算子，因为离散动作的选择隐含了代价。首次迭代可不加入，后续迭代若观察到无效动作频繁出现再考虑。

### 3.8 joint_condition_proxy
数学形式：`factor_1 * factor_2 * ...`（每个 factor 为连续 bounded 形式）或 `(f1 + f2 + ...) / n` 或 `(f1 * f2 * ...) ** (1/n)`
使用条件：没有显式 success flag，但有连续信号可构造任务完成的软近似。
适用证据：agent 能在各子条件分别取得进展但无法同时满足。
风险：乘积塌缩（一个 factor→0 则整体→0）；用几何平均或算术平均可缓解。

### 3.9 bounded_signal
数学形式：`x / (1 + abs(x))` 或 `1 / (1 + k * abs(error))` 或 `max(0, 1 - abs(error) / threshold)`
使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。用于压缩极端值而非施加约束。
与 hinge 的区别：bounded 是从两端压缩信号范围，hinge 是只在超出阈值时施加惩罚。如果目标是"值不应超过 X"，用 hinge；如果目标是"值不应该爆炸但无所谓具体范围"，用 bounded。

### 3.10 preview_conditioned_reward
数学形式：`main_reward * preview_factor`，preview_factor 基于观测中能反映**未来状态**的信号（如距离传感器、高度采样、前方地形探测），在不利前景下从 1 平滑衰减到下限。
使用条件：(1) 观测中存在提供前方/未来信息的维度；(2) 该维度可以映射到"前景好/坏"的连续度量；(3) agent 的失败模式与"无法提前调整行为以应对即将到来的状态变化"相关。
如何构造：从提供未来信息的观测中选择一个标量信号，设计一个在安全前景下接近 1、危险前景下接近下限（如 0.3-0.5）的衰减函数。下限不为零以避免完全抑制探索。
适用证据：agent 在相似的瞬时状态下表现差异大（同样的速度/姿态，有时成功有时失败），说明当前状态本身不足以区分好坏——缺少关于"接下来会发生什么"的信息。
与 soft_health_gate 的区别：gate 用当前的**身体状态**乘主奖励（"我已经歪了，别冲了"——被动响应）。preview 用**未来信息**乘主奖励（"前面是坑，别冲了"——主动预判）。两者可以共存：`main_reward * health_gate * preview_factor`。
风险：preview 信号若有噪声会导致主奖励波动；衰减下限设太低会抑制必要探索。

---


```
