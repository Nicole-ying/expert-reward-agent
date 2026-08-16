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
主体是一个 2D 简化的飞行器/着陆器，起始于视口顶部中央附近，并被赋予随机初始速度（随机力）。核心目标是**尽快到达画面中央的目标平台并稳定停靠**。过程中需尽可能少用推力（燃料优化），并保持稳定的姿态角度与低速着陆。该任务属于典型的**目标导向到达 + 软着陆控制**，不包含多目标冲突（燃料、速度、姿态均为从属优化）。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心是到达指定目标位置（中央平台）并停靠，附属目标为节省燃料、快速到达、姿态平稳，这些附属项不构成权重相当的冲突目标。动力学子类型进一步符合 goal_approach_and_soft_contact（接近目标并低速、稳定接触/停靠）。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: float32 (推测)  
- obs[0]: x_position – 相对目标平台的水平坐标，reward_usable: true  
- obs[1]: y_position – 相对平台高度的垂直坐标，reward_usable: true  
- obs[2]: x_velocity – 水平线速度，reward_usable: true  
- obs[3]: y_velocity – 垂直线速度，reward_usable: true  
- obs[4]: body_angle – 身体朝向角度，reward_usable: true  
- obs[5]: angular_velocity – 角速度，reward_usable: true  
- obs[6]: left_support_contact – 左支撑接触标志（1.0 表示接触），reward_usable: true  
- obs[7]: right_support_contact – 右支撑接触标志（1.0 表示接触），reward_usable: true  

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine – 不点燃任何引擎  
- action 1: left_orientation_engine – 点燃一个姿态引擎（顺时针或逆时针？推测会使身体旋转一个方向）  
- action 2: main_engine – 点燃主引擎（产生向上的推力）  
- action 3: right_orientation_engine – 点燃另一个姿态引擎（产生与 1 相反的旋转）  

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled – 身体进入“休眠”状态，通常表示已稳定停靠在平台且速度极低，可视为成功着陆。
- failure-like termination: crash_or_body_contact – 可能是身体撞击地面/山体/边界，或任何不应发生的接触；horizontal_position_outside_viewport – 水平位置超出视口范围。
- ambiguous termination: 以上两种 fate 的判断依赖环境内部物理逻辑，但从任务描述可以推断成功率由 body_not_awake_or_settled 定义，其余均为失败。
- truncation: 无明确最大步数截断，但一般由环境外部控制。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，没有 success 字段）
- explicit_failure_flag_available: false
- allowed_info_fields: {} （空字典，无任何额外字段）
- forbidden_or_uncertain_info_fields: info 中没有任何字段可用，因此不能依赖 info 去辨别成功/失败。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前观测（8维）
- action：当前步选择的动作（0-3）
- next_obs：下一步观测（8维）
- info：当前空字典，无可用字段
- training_progress：当前训练进度（0.0~1.0），仅当 prompt 明确允许时使用（此处未明确要求，故谨慎使用）

禁止使用：
- original_reward：官方奖励被遮蔽，不得使用
- 任何未在观测中声明的变量或 info 字段
- 任何基于官方奖励的重建或近似

## 7. 可用于奖励函数的信号
- position: obs[0] (x), obs[1] (y)，及其在 next_obs 中的对应值
- velocity: obs[2], obs[3]，next_obs[2], next_obs[3]
- orientation: obs[4]（角度）, obs[5]（角速度），next_obs 中亦然
- contact: obs[6]（左接触）, obs[7]（右接触），next_obs[6], next_obs[7]
- action/engine: 动作值（0:无, 1:左姿态, 2:主引擎, 3:右姿态），可用于能耗或推力惩罚
- other: 无其他可用信号

## 8. 不确定或不可用的信号
- 成功标志：没有显式 success/failure 标志，不能直接奖励“胜利”
- 平台中心绝对坐标：只有相对位置，无绝对目标坐标
- 燃料剩余/推力强度：未知，仅知道有 3 种引擎动作
- 碰撞细节：crash_or_body_contact 是什么物体，不可知

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body (2D)
  actuator_type: single_main_thruster + two_side_orientation_thrusters
  contact_structure: two_leg_support_contacts
primary_objectives:
  - arrive_at_target_pad_center
  - settle_with_low_velocity
secondary_objectives:
  - minimize_main_engine_usage
  - minimize_orientation_angle_and_angular_velocity
  - (implied) reach as fast as possible
main_failure_risks:
  - crashing_into_ground_or_boundary
  - overuse_of_thrust_leading_to_high_velocity_crash
  - losing_control_of_orientation_during_descent
  - failing_to_align_contacts_on_pad
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity  
  purpose: 鼓励主体向目标平台中心靠近。减少水平与垂直距离。  
  why_required: 任务是到达目标，必须驱动位置接近。  
  usable_signals: obs[0], obs[1] 或 next_obs[0], next_obs[1]  
  risks: 若仅依赖距离奖励，可能忽略减速和软着陆，导致高速坠毁。

- role_id: soft_contact_success  
  purpose: 奖励成功稳定停靠：双腿接触 + 速度极低。是实现“settled”终止的关键信号。  
  why_required: 任务明确要求停靠，仅靠距离到达无法保证任务完成。  
  usable_signals: next_obs[6], next_obs[7] (接触), next_obs[2], next_obs[3] (速度)  
  risks: 稀疏，早期难以获得；需要引入渐进奖励（如接近目标时接触奖励权重加大）。

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency  
  purpose: 惩罚不必要的燃料消耗，尤其避免无意义的主引擎持续喷射。  
  condition_to_use: 当 agent 尚未接近目标平台（例如距离较远）时，可使用轻微惩罚；当接近时，可加大惩罚以促进精细控制。  
  usable_signals: action（是否为 2）  
  risks: 过强惩罚会抑制探索，导致无法抵达；宜配合距离衰减权重。

- role_id: stability_penalty  
  purpose: 惩罚大的身体角度与角速度，促使姿态平稳着陆。  
  condition_to_use: 全局施加小权重；可在接近地面时（y 较小）增大权重，保证着陆阶段姿态稳定。  
  usable_signals: obs[4], obs[5] 或 next_obs[4], next_obs[5]  
  risks: 过大权重会妨碍必要的转向机动；需平衡。

- role_id: fast_approach  
  purpose: 奖励朝向目标的速度投影，满足“尽快到达”的从属目标。  
  condition_to_use: 只在未靠近地面且无接触风险时激活，例如 y > 某个阈值且速度未指向地面碰撞方向。  
  usable_signals: 利用当前位置方向与速度计算投影。  
  risks: 极易导致高速坠毁或 overshoot，必须配合减速约束。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: survival_balance_coherence  
  reason: 本任务无平衡存活需求，主体不会因倾斜而直接失败（除非撞击），因此不适用保持存活类职责。  
- role_id: direct_success_reward  
  reason: info 无成功标志，不能直接奖励成功。  
- role_id: action_smoothing  
  reason: 离散动作空间未提供平滑切换要求，且惩罚动作变化可能阻碍快速响应，不建议使用。  

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | next_obs[0], next_obs[1] | – | distance_to_target, dense_state_signal, bounded_signal | 可用欧氏距离或 -distance 作为密集奖励 |
| soft_contact_success | next_obs[6], next_obs[7], next_obs[2], next_obs[3] | info.success | product_of_contacts, threshold_logical, scaled_bonus | 当双腿接触且速度接近零时给予较大奖励 |
| energy_efficiency | action (==2) | fuel_consumption_value | binary_penalty, scaled_penalty | 每次点燃主引擎施加负奖励 |
| stability_penalty | next_obs[4], next_obs[5] | – | absolute_value, quadratic_penalty | 惩罚 abs(angle) 和 abs(angular_vel) |
| fast_approach | velocity projection toward target (using next_obs[0:3]) | – | projection_reward, conditional_gating | 需结合高度阈值与减速约束使用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 无法到达目标区域 | 平均最终距离大，奖励不上升 | 调整 goal_proximity 权重，或引入潜力塑形 |
| 高速坠毁/出界 | 最终速度高，平局终止原因多为 crash | 强化 soft_contact_success 和 stability_penalty，减小 fast_approach 权重 |
| 长时间盘旋不降落 | y 速度近零，x 抖动 | 增大接近时的 forward progress 奖励（谨慎），或利用接触标志触发 bonus |
| 过度使用主引擎 | 动作中 2 比例过高 | 增加 energy_efficiency 权重或引入渐进惩罚 |
| 姿态失控（角速度大） | 角度和角速度波动大 | 增大 stability_penalty 权重 |
| 单腿接触反复弹跳 | 接触标志频繁切换 | 奖励双腿同时接触且持续接触的状态，可加入持续时间阈值 |



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





# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -27.460

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| contact_bonus + goal_proximity + orientation_penalty + velocity_penalty | 3 | -27.460 | -27.460 | unsolved |
| goal_proximity + landing_progress + orientation_penalty + velocity_penalty | 1 | -111.620 | -111.620 | unsolved |
| crash_penalty + progress_reward + touchdown_bonus | 1 | -113.330 | -113.330 | unsolved |
| crash_penalty + landing_reward + progress_reward | 2 | -307.970 | -307.970 | unsolved |

## Previous interventions

- iter 4 (score=-111.620, structure=goal_proximity + landing_progress + orientation_penalty + velocity_penalty): selected_level**：**L2: 理由：需要改变 contact_bonus 的数学形态——从稠密状态奖励改为“接触事件 + 一次性成功着陆事件”，属于 L2 语义修改。权重会适当微调，但核心是公式形态变化。 | selected_intervention**：
- iter 7 (score=-113.330, structure=crash_penalty + progress_reward + touchdown_bonus): `selected_level`：**L2: 证据表明当前的主信号框架（持续 proxy）被 exploit，必须改变数学形态：移除持续的 `landing_proxy`，替换为一次性接触事件奖励。这不是简单调参（L1），而是语义级别的组件替换，但又未到全盘推倒主框架（L3），因为接近目标的方向信号（progress）仍然有价值。 | `selected_intervention`：

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.

```
