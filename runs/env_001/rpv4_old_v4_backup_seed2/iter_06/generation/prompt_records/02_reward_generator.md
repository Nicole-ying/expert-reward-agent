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
主目标：控制一个具有主引擎和左右方向引擎的2D飞行器，从起始区域（大致上方正中）开始，尽可能快速地降落到视野中央的着陆平板上，并实现稳定停靠（姿态平稳、速度接近于零、双脚接触标志均为真）。  
次目标：在完成主目标的前提下，尽可能减小引擎推力使用量（即降低燃料消耗）。  
不可混淆的目标：单纯存活、无目的悬停或长时间飞行都不符合任务意图，高油耗不应作为唯一优化方向。

## 2. 任务类型选择
- selected_route_id: **navigation_goal_reaching**
- confidence: **high**
- reason: 核心目标是到达指定空间位置（中央着陆平板）并安全停靠，属于典型的“导航目标到达”任务。省燃料、快速到达等附属目标不影响主分类。

## 3. 观察空间 observation_space
- type: Box（连续向量）
- shape: [8]
- dtype: float32
- 各维度含义：

| 索引 | 字段名 | 含义 | reward_usable |
|------|--------|------|---------------|
| 0 | x_position | 飞行器相对着陆平板中心的水平坐标 | true |
| 1 | y_position | 飞行器相对着陆平板顶部高度的垂直坐标 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 飞行器倾斜角度（弧度） | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左脚接触标志（1.0 接触，0.0 未接触） | true |
| 7 | right_support_contact | 右脚接触标志（1.0 接触，0.0 未接触） | true |

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：

| 动作编号 | 名称 | 含义 |
|----------|------|------|
| 0 | no_engine | 无任何引擎点火，自由运动 |
| 1 | left_orientation_engine | 点火左侧方向引擎（可能产生逆时针力矩或左向推力） |
| 2 | main_engine | 点火主引擎（通常产生向上的推力） |
| 3 | right_orientation_engine | 点火右侧方向引擎（可能产生顺时针力矩或右向推力） |

说明：左右方向引擎用于调整飞行器姿态，主引擎主要用于减速/悬停，具体物理效果由掩码动力学决定。

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` —— 飞行器稳定停靠（通常意味着降落在平板上且速度、角速度都趋于零）。此时双脚接触标志很可能都为1。
- **failure-like termination**:  
  - `crash_or_body_contact` —— 身体非正常碰撞（例如与地板强烈撞击、侧翻导致其他部位接触），可能包含未落在平板上的情况。  
  - `horizontal_position_outside_viewport` —— 水平位置超出视野边界，失控飞出。
- **ambiguous termination**: 若 `body_not_awake_or_settled` 发生在非平板区域（理论上不会，因为接触标志可能为0），但目前信息无法完全排除。
- **truncation**: 作业描述中未提及 episode 截断步数，但通常 RL 环境存在默认步数上限，此处未声明，暂不考虑。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: **false**（info 返回空字典）
- **explicit_failure_flag_available**: **false**
- allowed_info_fields: {}（无可用字段）
- forbidden_or_uncertain_info_fields: 任何 info 内的字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

**允许使用**：
- `obs`（完整 8 维向量，含义见上）
- `action`（0~3 整数）
- `next_obs`（下一时刻的 8 维向量）
- `info` 中明确声明可用的字段（目前无任何字段可用）
- `training_progress` 仅当 prompt 明确允许时可用（本任务并未允许，因此**禁止使用**）

**严格禁止**：
- `original_reward`（官方奖励已被掩码，不可参考）
- 任何未声明的 `info` 字段（如 `info['success']` 等）
- 任何未列出的 `obs` 切片假设

## 7. 可用于奖励函数的信号
从 `obs` 和 `next_obs` 中可提取：
- **位置**：x_position, y_position（相对平板中心的高度差）
- **速度**：x_velocity, y_velocity
- **姿态**：body_angle, angular_velocity
- **接触**：left_support_contact, right_support_contact（0或1）
- **动作/推力**：action 编号（可间接计算是否使用了引擎）
- **派生量**：可计算到目标的水平距离 sqrt(x_pos² + y_pos²)，速度范数，是否双脚全部接触（contact_all = left*right），是否接近静止等。

## 8. 不确定或不可用的信号
- 终止原因无法从 info 获取，reward 函数内无法获知 episode 是否已终止（除非额外传入 terminated 标志，但当前接口不包含）
- 无法区分“crash”与“settle”终止信号，只能通过状态（接触、速度、角度）事后推断
- 无法获取步长限制信息
- 无法获取当前环境真实几何尺寸（如平板宽度、高度限制），依靠经验调节阈值

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D_lander (主引擎 + 两侧方向姿态引擎)
  actuator_type: thruster (离散开关)
  contact_structure: two_legs_with_contact_indicators
primary_objectives:
  - 准确降落到中央着陆平板中心
  - 实现双脚稳定接触并静止 (速度和角速度趋于零)
secondary_objectives:
  - 最小化引擎使用次数（降低燃料消耗）
  - 缩短降落飞行时间（隐含快速性）
main_failure_risks:
  - 水平飞出视野边界
  - 剧烈碰撞导致 crash（可能因高速或侧翻）
  - 在平板外区域静止（非成功停靠）
  - 过度使用引擎导致无法收敛到节能策略
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- **role_id**: approach_to_target
  - purpose: 鼓励飞行器不断缩小与着陆平板中心的距离
  - why_required: 必须驱动 agent 向目标位置移动，是任务最基础的前进方向
  - usable_signals: [x_position, y_position] （可直接构造 距离惩罚 = sqrt(x² + y²) 或分段）
  - risks: 距离惩罚过重可能导致高速冲撞，需与减速职责配合

- **role_id**: soft_landing_velocity_control
  - purpose: 当飞行器接近平板时迫使速度减小，实现软着陆
  - why_required: 单纯接近目标可能导致高速撞击，必须抑制速度
  - usable_signals: [x_velocity, y_velocity]
  - risks: 阈值设置不当会抑制正常下降，或产生过多控制震荡

- **role_id**: stable_contact_termination_bonus
  - purpose: 当检测到双脚均接触、速度/角速度接近于零时，给予较大的成功奖励
  - why_required: 提供最终目标完成的稀疏强化信号，否则不易学会精确停靠
  - usable_signals: [left_support_contact, right_support_contact, x_velocity, y_velocity, body_angle, angular_velocity]
  - risks: 如果奖励过于稀疏可能学习缓慢，如果阈值过于宽松可能误判

- **role_id**: orientation_stabilization
  - purpose: 在接近至平板附近时，惩罚过大倾斜角度和角速度，防止侧翻 crash
  - why_required: 姿态失控会导致快速失败，必须在降落阶段维持平衡
  - usable_signals: [body_angle, angular_velocity]
  - risks: 远离平板时不必强制姿态修正，须加入距离条件以免过度约束

- **role_id**: fuel_efficiency_penalty
  - purpose: 对主引擎和方向引擎的使用进行轻微惩罚，促使节省燃料
  - why_required: 任务明确要求“使用尽可能小的引擎推力”，不加入会学到滥推
  - usable_signals: [action]（非零动作即表示点火）
  - risks: 惩罚权重过高会导致 agent 不愿点火，完全放弃控制；需组合其他职责平衡

### 10.2 条件职责 conditional_roles
- **role_id**: speed_modulation_in_freefall
  - condition_to_use: 当飞行器距离平板较远（下降初期），可对水平速度持续惩罚以维持大致垂直下落，但非必选，可由主职责覆盖
  - usable_signals: [x_velocity]
  - risks: 可能抑制必要的横向移动

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: survival_time_reward
  - reason: 任务目标是尽快降落到平板，而非长时间存活，无生存奖励信息支持
  - forbidden_or_missing_signals: [episode_step, alive_flag]
- **role_id**: exploration_bonus
  - reason: 状态空间连续且目标明确，不需要刻意探索奖励
- **role_id**: multi_objective_energy_distance_balance
  - reason: 虽有多指标，但主次分明，无需多目标权衡框架

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_to_target | `x_position`, `y_position` | none | `euclidean_distance`, `negative_distance`, `quadratic_penalty` | 距离越近奖励越大 |
| soft_landing_velocity_control | `x_velocity`, `y_velocity` | none | `velocity_norm`, `quadratic_penalty`, `distance_thresholded_scale` | 只有靠近平板时才有效 |
| stable_contact_termination_bonus | `left_contact`, `right_contact`, `x_velocity`, `y_velocity`, `body_angle`, `angular_velocity` | none | `all_contacts_true`, `norm_small`, `step_function_bonus` | 条件：双脚接触 and 速度角速度极小 |
| orientation_stabilization | `body_angle`, `angular_velocity` | none | `absolute_value`, `quadratic_penalty`, `distance_gated` | 加以距离门控避免过早约束 |
| fuel_efficiency_penalty | `action` | none | `indicator_nonzero` | 惩罚任何非零引擎动作 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 水平飞出视野边界 | episode 提前终止，x_position 绝对值持续增大 | 加强横向速度惩罚，或增加目标方向引导奖励 |
| 高速撞击平板导致 crash | 接触时速度很大，双脚接触可能为1但 reward 很低（因 crash终止） | 施加速度惩罚并提高软着陆奖励的阈值灵敏度 |
| 在平板边缘悬停或滑动但不停靠 | 接触标志部分为1，速度不为零，episode 不停（被截断） | 增加成功接触的严格条件（双脚接触+极低速度），提高成功奖励密度 |
| 过度翻滚（body_angle 较大）后 crash | 角速度或角度持续增大 | 增加姿态稳定惩罚的权重或提前触发条件 |
| 燃料消耗过大（不停点火） | 每个动作点火的频率极高，但可能由于缺乏惩罚而不学节能 | 启动燃料效率惩罚，但注意平衡以避免不动作导致任务失败 |



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
- best_score_so_far: -37.030

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| angular_vel_penalty + landing + progress + speed_penalty | 4 | -37.030 | -197.460 | unsolved |
| angular_vel_penalty + progress + speed_penalty + velocity_guidance | 1 | -295.330 | -295.330 | unsolved |

## Previous interventions

- iter 2 (score=-37.030, structure=angular_vel_penalty + landing + progress + speed_penalty): 现在我已经掌握了完整的画面。让我整理诊断并制定修改方案。 | selected_level: L2** — 修改 `landing` 组件的数学形态：从"严格乘积门控（5 个因子全部非零才有奖励）"改为"基于 proximity 门控的连续求和公式，提供部分梯度"。这不是简单调权重（L1），也不是换框架（L3）。 | selected_intervention: 目标组件**：`landing`（着陆奖励）
- iter 3 (score=-222.850, structure=angular_vel_penalty + landing + progress + speed_penalty): 现在我已掌握完整画面。以下是诊断和修改方案。 | selected_level: L2** — 需要改变 landing 的数学形态（从 proximity-gated 改为 contact-gated），单纯调权重（L1）无法解决根本逻辑缺陷：无论权重多低，只要 proximity > 0 就给 reward，agent 就会 exploit。 | selected_intervention: 目标组件**：`landing`
- iter 4 (score=-295.330, structure=angular_vel_penalty + progress + speed_penalty + velocity_guidance): 现在我已经掌握了所有关键信息。让我整理诊断并制定修改方案。 | selected_level: L2**：替换一个组件的数学形态/语义。不是调系数（L1），也不是全框架重构（L3）。 | selected_intervention: 目标组件：** 移除 `landing`，新增 `velocity_guidance`。
- iter 5 (score=-197.460, structure=angular_vel_penalty + landing + progress + speed_penalty): selected_level: L2** — 替换一个组件的数学形态。将 `velocity_guidance`（方向对齐）替换为 `landing`（proximity-gated + contact-amplified 着陆引导）。这不是调参（L1），也不是换整个框架（L3）。这是将"方向引导"的语义改为"目标接近+着陆质量引导"。 | selected_intervention: 移除 `velocity_guidance`，新增 `landing` 组件**：

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.

```
