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
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

## 2. 任务类型选择
- **selected_route_id**: navigation_goal_reaching
- **confidence**: high
- **reason**: 任务的最终成功明确依赖于“到达并稳定在目标垫板上”，属于典型的带软着陆的导航目标到达，节油和快速只是附属优化，不构成冲突的多目标问题。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: float32（典型）
- **各维度含义与奖励可用性**：
  - `obs[0]`: **x_position** – 相对于目标垫板的水平坐标，单位未知。奖励可用：是。
  - `obs[1]`: **y_position** – 相对于垫板高度的垂直坐标（越接近0表示越靠近垫板表面）。奖励可用：是。
  - `obs[2]`: **x_velocity** – 水平线速度。奖励可用：是。
  - `obs[3]`: **y_velocity** – 垂直线速度。奖励可用：是。
  - `obs[4]`: **body_angle** – 机体朝向角度。奖励可用：是。
  - `obs[5]`: **angular_velocity** – 角速度。奖励可用：是。
  - `obs[6]`: **left_support_contact** – 左支撑腿是否接触地面的二进制标志（1.0/0.0）。奖励可用：是。
  - `obs[7]`: **right_support_contact** – 右支撑腿是否接触地面的二进制标志（1.0/0.0）。奖励可用：是。

所有8维观测均可用于奖励函数设计。

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **动作含义**：
  - `0` (no_engine)：无推力输出，仅靠惯性运动。
  - `1` (left_orientation_engine)：点燃左侧姿态引擎，产生力矩使机体逆时针旋转（或按物理定义）。
  - `2` (main_engine)：点燃主引擎，提供正Y方向的推力（向上的力，用于减速下降/调整高度）。
  - `3` (right_orientation_engine)：点燃右侧姿态引擎，产生反向力矩，使机体顺时针旋转。

动作离散，适合进行使用惩罚（fuel efficiency），但无法直接获得推力大小的连续信号。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success‑like termination**:
  - `body_not_awake_or_settled`：机体进入休眠或判定为已稳定（速度、角速度极低并可能接触垫板）。这通常表示安全着陆任务完成。
- **failure‑like termination**:
  - `crash_or_body_contact`：机体的主体部分或非支撑腿部位发生碰撞，视为坠毁。
  - `horizontal_position_outside_viewport`：水平位置完全离开可见区域，相当于飞出任务区。
- **ambiguous termination**:
  - （无）
- **truncation**:
  - 所提供的 `step` 源码中 `truncated` 固定返回 `False`，因此没有时间截断，但实际包装可能隐含有最大步数，此处不作假设。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false（`info` 字典为空，无法直接获得成功/失败标志）。
- **explicit_failure_flag_available**: false。
- **allowed_info_fields**: 无（`info` 始终为 `{}`）。
- **forbidden_or_uncertain_info_fields**: 所有未声明的字段均禁止使用，特别是 `success`、`termination_reason`、`failure` 等。

## 6. reward 函数接口契约
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用**：
- `obs` – 当前观测（8维）
- `action` – 当前动作（离散 0‑3）
- `next_obs` – 下一观测（8维）
- `info` – 只允许访问明确声明的字段（当前无）
- `training_progress` – 仅当后续 prompt 明确声明可用时才允许使用

**禁止使用**：
- `original_reward`
- 任何未声明的 `info` 字段
- 任何未文档化的观测切片或假设

## 7. 可用于奖励函数的信号
- **位置相关**：`x_position`、`y_position` – 可直接测量与目标垫板的水平和垂直偏差。
- **速度相关**：`x_velocity`、`y_velocity` – 衡量是否接近静止。
- **姿态相关**：`body_angle`、`angular_velocity` – 衡量机体是否水平稳定。
- **接触相关**：`left_support_contact`、`right_support_contact` – 判断是否双支撑安全落地。
- **动作/引擎**：`action` – 离散动作，可用于计算推力消耗（action ≠ 0 视为消耗燃料）。
- **其他**：无。

## 8. 不确定或不可用的信号
- 无明确的成功/失败标志（`info` 为空）。
- 无碰撞强度或接触力测量。
- 无剩余燃料量。
- 无精确时间步计数。
- 无法获得风扰或其他外部干扰的显式信号。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D vehicle-like body with bilateral landing legs
  actuator_type: discrete thrusters (main engine + two orientation engines)
  contact_structure: two support pads (left/right) that can contact the ground
primary_objectives:
  - reach the target pad (minimize x, y deviation)
  - settle stably (low velocity, low angular velocity, upright orientation)
  - achieve safe two-leg contact on the pad
secondary_objectives:
  - time-to-target (as fast as possible)
  - fuel saving (minimize engine usage)
main_failure_risks:
  - crashing with body parts other than legs
  - drifting out of the horizontal viewport
  - excessive oscillations that prevent settling
  - premature contact without controlled approach (hard landing)
  - fuel waste from constant thruster usage
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: goal_proximity**
  - purpose: 驱使机体向目标垫板靠近，并最终贴齐垫板表面。
  - why_required: 到达目标是任务根本，必须将位置误差驱动至零。
  - usable_signals: `x_position`, `y_position`, `x_velocity`, `y_velocity`
  - risks: 若纯用位置惩罚，可能导致高速冲击目标；需配合稳定性约束。

- **role_id: settling_stability**
  - purpose: 抑制接近目标时的速度、角速度和不必要摇摆，保证机体能够平稳降落。
  - why_required: 不稳定的接近会导致接触失败或触发碰撞终止。
  - usable_signals: `x_velocity`, `y_velocity`, `angular_velocity`, `body_angle`
  - risks: 若强度过高，可能阻止快速接近，需要在接近目标时逐渐加强。

### 10.2 条件职责 conditional_roles
- **role_id: safe_landing_contact**
  - condition_to_use: 当机体已充分接近目标垫板（如 `|x_position|<threshold` 且 `|y_position|<threshold`）且速度较低时激活。
  - purpose: 鼓励双支撑腿同时接触垫板，完成“软着陆”。
  - usable_signals: `left_support_contact`, `right_support_contact`
  - risks: 无条件激活可能导致 agent 在远处故意摔腿或撞击地面，必须配合 gate 条件。

- **role_id: fuel_efficiency**
  - condition_to_use: 作为轻度惩罚持续存在，但不应抑制必要的探索。可在训练后期增大权重。
  - purpose: 最小化引擎使用次数（每个非零动作计为消耗）。
  - usable_signals: `action`
  - risks: 过强的燃料惩罚会使 agent 完全不点火，无法完成导航。

- **role_id: upright_orientation**
  - condition_to_use: 仅在机体接近目标或速度较小时激活。
  - purpose: 保持机体角度接近 0，避免倾斜着陆。
  - usable_signals: `body_angle`
  - risks: 若全程强调角度，可能干扰大角度机动，减慢学习。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: explicit_success_bonus**
  - reason: 环境中没有可访问的 success flag，终止原因不可用于奖励函数，强行使用会导致不可知。
  - forbidden_or_missing_signals: `info["success"]` (不存在)。

- **role_id: sparse_reward**
  - reason: 奖励函数无法获取终止信息，仅靠终止事件作为稀疏奖励不可行；任务需要密集引导信号。
  - forbidden_or_missing_signals: 结束标志不可访问。

- **role_id: collision_penalty**
  - reason: 没有碰撞强度或碰撞事件的可观测信号，唯一碰撞感知来自终止，不能用于奖励内部。
  - forbidden_or_missing_signals: 碰撞传感器不存在。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|--------|----------------|----------------|----------------------------|-------|
| goal_proximity | `x_position`, `y_position`, `x_velocity`, `y_velocity` | - | `dense_state_signal`, `distance_to_target`, `velocity_to_target` | 组合位置与速度的 progress 奖励；可单独对横向/纵向偏差加权。 |
| settling_stability | `x_velocity`, `y_velocity`, `angular_velocity`, `body_angle` | - | `bounded_signal`, `quadratic_penalty`, `weighted_sum` | 速度惩罚可用平方或门控，接近目标时权重升高。 |
| safe_landing_contact | `left_support_contact`, `right_support_contact` | - | `gated_signal`, `(contact_left AND contact_right) * gate_progress` | 需与进度或低速门结合，防止误奖励。 |
| fuel_efficiency | `action` | exact thrust magnitude (离散动作) | `discrete_action_penalty`, `(action != 0) * penalty` | 轻微负奖励，可进行调度（如 curriculum）。 |
| upright_orientation | `body_angle` | - | `bounded_signal`, `abs_penalty` | 可配合接触条件使用，避免倒立着陆。 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|-------------|-------------------|----------------------|
| **hover without approaching** | `y_position` 不下降，`y_velocity ≈ 0`，agent 不点火主引擎或用错方向 | 增强 goal_proximity 的 y 方向权重，减小燃料惩罚，引入高度下降的 increment reward |
| **fly out of horizontal bounds** | `x_position` 持续增大，超出视口导致终止 | 增加横向偏差的惩罚，检查 orientation roles 是否过度鼓励旋转，可能需要姿态限幅 |
| **crash at high speed** | `|y_velocity|` 或 `|x_velocity|` 在接触前仍然很大，接触后立即 crash | 加强 settling_stability（尤其 y_velocity），在接近垫板时使用指数级增强的速度惩罚 |
| **waste excessive fuel** | 动作日志中非零动作频率极高，但位置无明显改良 | 适当增加 fuel_efficiency 惩罚，或采用“只有无动作时奖励”的保守策略；可能需配合探索 |
| **oscillate without settling** | `body_angle` 或 `x_position` 在垫板附近来回摆动，无法触发 body_not_awake | 提升稳定性的角速度/角度的惩罚权重，或将 settling 相关奖励设为在低速度区域内才最大化 |
| **always no‑op and fall** | agent 学习为 0 动作以避免燃料惩罚，但直接掉落导致 crash | 动态调整 fuel_efficiency 惩罚强度，确保 survival 相关奖励（如下降速度控制）远大于燃料惩罚 |



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
