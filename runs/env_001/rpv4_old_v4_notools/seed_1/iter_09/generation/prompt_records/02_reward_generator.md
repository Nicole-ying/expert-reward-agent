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
控制一台2D飞行器（带主发动机和两侧姿态发动机），从视口顶部中央区域出发，尽快零速平稳降落到中央目标平台，同时尽量少用推进燃料。学习策略需兼顾以下子目标：
- 快速抵近目标平台（x,y 接近零）；
- 下降触地时垂向和水平速度足够低（安全接触）；
- 着陆时保持机身姿态接近水平（角度≈0）；
- 累计推力使用量越小越好。

> 注意：不要将“节省燃料”独立为一个同等优先级目标；它是抵达并安全着陆后的优化指标。

## 2. 任务类型选择
selected_route_id: `navigation_goal_reaching`  
confidence: `high`  
reason: 核心诉求是使飞行器到达目标位置并稳定着陆，属于典型的“到达目标点”导航任务；附属目标（省燃料、快速到达）不改变任务主类型。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64 (从示例看)
- 各维度含义与可用性：
  - obs[0] (x_position)：相对目标平台的水平偏移，`reward_usable: true`
  - obs[1] (y_position)：相对平台高度的垂直偏移，`reward_usable: true`
  - obs[2] (x_velocity)：水平线速度，`reward_usable: true`
  - obs[3] (y_velocity)：垂直线速度，`reward_usable: true`
  - obs[4] (body_angle)：机身俯仰角，`reward_usable: true`
  - obs[5] (angular_velocity)：角速度，`reward_usable: true`
  - obs[6] (left_support_contact)：左支撑腿接触标志（0/1），`reward_usable: true`
  - obs[7] (right_support_contact)：右支撑腿接触标志（0/1），`reward_usable: true`

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 各动作含义：
  - action 0: `no_engine` – 不点火
  - action 1: `left_orientation_engine` – 左侧姿态发动机点火（产生逆时针力矩 + 可能的小横向推力）
  - action 2: `main_engine` – 主发动机点火（产生向上推力 + 可能的旋转影响）
  - action 3: `right_orientation_engine` – 右侧姿态发动机点火（顺时针力矩 + 可能的小横向推力）

## 5. step 与终止条件分析
### 5.1 终止模式
根据提供的终止条件：
- `crash_or_body_contact`：可能为失败（坠毁或危险接触）
- `horizontal_position_outside_viewport`：失败（飞出视野）
- `body_not_awake_or_settled`：身体稳定停住（极可能是成功着陆，即成功终止）

此外，还存在隐式截断（步数上限），但未在任务描述中说明。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: `false` （无 info["success"] 等显式标记）
- explicit_failure_flag_available: `false`
- allowed_info_fields: 空的 `{}`，本环境无可用 info 字段
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（为空）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs`：当前与下一时刻的观测均可使用（`obs`, `next_obs`）
- `action`：当前执行的动作
- `info`：仅内容为空，不可使用任何字段
- `training_progress`：只有明确提示此参数可用时才利用

禁止使用：
- `original_reward`（已屏蔽的官方奖励）
- 未在 `obs`、`action`、`next_obs` 中定义的任何信号
- 任何未明确允许的 info 字段

## 7. 可用于奖励函数的信号
- 位置相关：`x_position (obs[0]/next_obs[0])`、`y_position (obs[1]/next_obs[1])`，可计算目标距离
- 速度相关：`x_velocity (obs[2]/next_obs[2])`、`y_velocity (obs[3]/next_obs[3])`，可用于接触缓速
- 姿态相关：`body_angle (obs[4]/next_obs[4])`、`angular_velocity (obs[5]/next_obs[5])`
- 接触标志：`left_support_contact (obs[6]/next_obs[6])`、`right_support_contact (obs[7]/next_obs[7])`
- 动作：`action (0-3)`，可将推力使用量作为惩罚
- 其他：`next_obs` 可用于方向性变化等

## 8. 不确定或不可用的信号
- 显式成功/失败标记：不存在，不可用
- 与平台精确对齐的额外信息（如平台尺寸）：无
- 剩余燃料或速度上限：无
- 脱离视口的精确边界值：未提供（已知有水平边界，但缺数值，不建议依赖）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 单一刚体（带两条支撑腿）
  actuator_type: 推力器（主发动机 + 两姿态推进器）
  contact_structure: 两个支撑点（左/右）
primary_objectives:
  - 将飞行器中心降至目标平台（x,y ≈0）
  - 以几乎零速、机身水平状态接触并停稳
secondary_objectives:
  - 最小化燃料消耗（减少非零动作）
  - 尽快完成降落（时间效率隐含在燃料/推进选择中）
main_failure_risks:
  - 撞击或高速接触导致坠毁（crash）
  - 水平飞出执行区
  - 悬停/不降落导致截断
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: `goal_proximity`
  purpose: 鼓励飞行器位置向目标点移动，距离越小奖励越高
  why_required: 核心导航目标，缺少则无法引导到达目标
  usable_signals: `[x_position, y_position] from obs/next_obs`
  risks: 若仅用直线距离可能导致不合理的路径（如直接高速俯冲）；需配合速度约束

- role_id: `soft_landing`
  purpose: 在接触或接近平台时，惩罚过大的垂向和水平速度
  why_required: 任务要求安全接触，高速着陆等同失败
  usable_signals: `[x_velocity, y_velocity] from next_obs，以及接触标志 left/right_support_contact`
  risks: 过早惩罚速度可能抑制正常飞行；适合在接近地面时激活

- role_id: `upright_stabilization`
  purpose: 保持 body_angle 接近 0，并抑制角速度
  why_required: 着陆时机身倾斜可能导致倾倒或坠毁
  usable_signals: `[body_angle, angular_velocity] from obs/next_obs`
  risks: 在远离平台时过度要求直立可能妨碍机动

- role_id: `throttle_efficiency`
  purpose: 对使用任何发动机的动作施加惩罚
  why_required: 任务明确要求“as little engine thrust as possible”
  usable_signals: `action`（0 为无推力，其他为推力动作）
  risks: 若惩罚过重可能导致不敢下降；需与其他职责平衡

### 10.2 条件职责 conditional_roles
- role_id: `terminal_settle_boost`
  condition_to_use: 当接触标志显示已着陆（left+right contact=1）且速度接近零时，可给予额外成功信号（如正向奖励），但需谨慎因无显式成功标记
  usable_signals: `[left_support_contact, right_support_contact] & [x_velocity, y_velocity] from next_obs`
  risks: 可能意外奖励非成功停稳状态；强烈建议在训练后期引入或加权极小

### 10.3 慎用/禁用职责 avoid_roles
- role_id: `success_flag_boost`
  reason: 环境中不存在可检测的全局 success/failure 标记，info 为空，无法实现基于官方判断的奖励。
  forbidden_or_missing_signals: `info['success']`, `info['failure']`, `terminated` 原因等

- role_id: `crash_penalty_via_termination`
  reason: 无法在奖励函数中同步获取终止理由，无法直接实现“撞毁扣分”，只能通过速度/接触状态近似。
  forbidden_or_missing_signals: 显式 crash 信号

## 11. role_to_signal_mapping
| role_id                | usable signals                                      | missing signals               | candidate formula operators          | notes |
|------------------------|-----------------------------------------------------|-------------------------------|--------------------------------------|-------|
| `goal_proximity`       | x_position, y_position (obs/next_obs)               | 无                            | dense_state_signal, bounded_distance | 应使用平滑递减（如负距离或高斯） |
| `soft_landing`         | x_velocity, y_velocity, left/right contact (next_obs) | 精确接触力/冲击大小          | velocity_penalty, conditional_scale   | 建议在 y 接近 0 且接触发生前后加大权重 |
| `upright_stabilization`| body_angle, angular_velocity (obs/next_obs)           | 无                            | angle_cost (abs/quadratic), angular_velocity_cost | 可合并为单一姿态误差 |
| `throttle_efficiency`  | action (0-3)                                        | 推力大小（只有开/关）        | action_cost (sparse penalty per nonzero action) | 非零动作均记为消耗 |
| `terminal_settle_boost`| left/right contact + velocities (next_obs)          | 稳健的成功检测               | conditional_bonus (small fixed)       | 极易误判，慎用 |


## 12. 初始训练后应观察的 failure modes
| failure_mode          | evidence_to_check                                              | possible_intervention                            |
|-----------------------|----------------------------------------------------------------|--------------------------------------------------|
| 长时间悬空不下落      | episode长度接近上限，y_position 始终偏高                       | 引导 reward 中加入高度惩罚，或提高主发动机推力使用容忍度 |
| 高速撞击平台          | 终止时 y_velocity 较大，且 crash 终止                          | 强化 soft_landing 角色，调大速度惩罚权重          |
| 姿态失控翻转          | body_angle 或 angular_velocity 持续过大                        | 增强 upright_stabilization，限制角速度           |
| 过量侧移飞出视口      | x_position 超出边界，horizontal_position_outside_viewport      | 增大 horizontal 偏离惩罚，或奖励接近中心线       |
| 极少使用发动机导致未能纠正坠落 | 动作几乎全是 0，y_velocity 持续负增长              | 平衡 throttle_efficiency 和 goal_proximity 权重，避免惩罚过度 |
| 着陆后仍不终止        | 接触标志为 1 但速度非零或微小滑动导致“settled”未触发         | 检查接触后速度，或在接触后降低 soft_landing 权重避免悬停奖励 |



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
- best_score_so_far: 39.440

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| attitude_penalty + proximity_progress + soft_landing_bonus | 4 | 39.440 | 39.440 | unsolved |
| attitude_penalty + landing_quality_reward + proximity_progress + speed_penalty + touchdown_reward | 1 | -24.390 | -24.390 | unsolved |
| attitude_penalty + leg_contact_reward + proximity_progress + soft_landing_bonus + speed_penalty | 1 | -39.690 | -39.690 | unsolved |
| attitude_penalty + proximity_progress + soft_landing_bonus + speed_penalty + touchdown_reward | 1 | -51.020 | -51.020 | unsolved |
| attitude_penalty + proximity_progress + soft_landing_bonus + speed_penalty | 1 | -108.890 | -108.890 | unsolved |

## Previous interventions

- iter 4 (score=39.440, structure=attitude_penalty + proximity_progress + soft_landing_bonus): selected_level**：L2: 不能仅靠 L1 调系数解决 exploit，必须改变 `soft_landing_bonus` 的数学形态，从持续过程奖励变为首次着陆事件的一次性奖励，并嵌入位置精度因子。 | selected_intervention: 目标组件 `soft_landing_bonus`：
- iter 5 (score=-108.890, structure=attitude_penalty + proximity_progress + soft_landing_bonus + speed_penalty): selected_level: L2。需要修改 `soft_landing_bonus` 的数学形态（从可重复触发的事件改为一次性成功事件），并联动新增一个速度惩罚组件（服务于同一语义——引导减速、杜绝弹跳刷分）。 | selected_intervention: 1. 将 `soft_landing_bonus` 重构为 **成功着陆一次性奖励**：仅在从“不满足成功条件”到“满足成功条件”的转变时给予 200 奖励。成功条件严格：双腿接触、位置足够近、速度足够低、姿态足够平。
- iter 6 (score=-39.690, structure=attitude_penalty + leg_contact_reward + proximity_progress + soft_landing_bonus + speed_penalty): selected_level**：L2 —— 需要引入一个新的连续奖励组件来填补信号空缺，使腿接触行为得到引导。 | selected_intervention**：
- iter 7 (score=-51.020, structure=attitude_penalty + proximity_progress + soft_landing_bonus + speed_penalty + touchdown_reward): selected_level: L2 – 修改 leg_contact_reward 的数学形态，将其替换为基于着陆质量的连续奖励，同时微调 proximity_progress 权重以平衡，整体服务于同一语义：为着陆阶段提供精细梯度信号。 | selected_intervention: 删除原 `leg_contact_reward`，新增 `touchdown_reward`：只在有腿接触时，以 `(1 - dist/0.3) * (1 - speed/0.3) * (1 - angle/0.2)` 刻画着落质量，系数 10.0，clip 到 [0, 1] 再乘系数，避免负值。
- iter 8 (score=-24.390, structure=attitude_penalty + landing_quality_reward + proximity_progress + speed_penalty + touchdown_reward): selected_level: L2** — 需要修改数学形态，而不仅仅是调参。稀疏事件奖励改为密集每步奖励、速度惩罚引入阶段门控、touchdown quality 从乘积改为加和，这些都不是单纯系数调整。 | selected_intervention: 1. **移除 `soft_landing_bonus`**，替换为 `landing_quality_reward`——成功条件满足时每步给 +10，持续奖励稳定着陆，消除大额脉冲。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.

```
