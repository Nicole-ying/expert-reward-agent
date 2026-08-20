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
训练一个 3D 四足机器人，通过扭矩控制 8 个关节实现稳定、快速的前向行走或奔跑，并始终保持身体高度在健康范围内 (0.2 ~ 1.0)。**核心目标**是持续前向运动，而不是单纯站立不倒；次要目标包括保持直立姿态、抑制侧向漂移、降低能耗并避免翻倒或崩溃。

## 2. 任务类型选择
**selected_route_id**: `locomotion_continuous_control`  
**confidence**: high  
**reason**: 任务核心是让四足机器人沿着一个方向（默认世界 x 轴）持续前进，附属有高度稳定、姿态保持等约束，符合“持续前进通过地形”的定义，不涉及目标点到达，也不仅仅为了平衡存活。

## 3. 观察空间 observation_space
- **type**: `Box`  
- **shape**: `[27]`  
- **dtype**: 未显式声明，通常为 `float32`（连续动力学环境）  
- 各维度含义及可用于奖励的说明：

| 索引 (index) | 名称 | 含义 | reward_usable |
|---|---|---|---|
| 0 | body_z | 身体中心垂直高度 | true（高度健康约束） |
| 1 | quat_w | 身体朝向四元数 w | true（用于计算 body_up_z） |
| 2 | quat_x | 身体朝向四元数 x | true |
| 3 | quat_y | 身体朝向四元数 y | true |
| 4 | quat_z | 身体朝向四元数 z | true |
| 5 | joint_1_angle | 第 1 髋关节角度 | true（关节姿态平滑/约束） |
| 6 | joint_2_angle | 第 1 踝关节角度 | true |
| 7 | joint_3_angle | 第 2 髋关节角度 | true |
| 8 | joint_4_angle | 第 2 踝关节角度 | true |
| 9 | joint_5_angle | 第 3 髋关节角度 | true |
| 10 | joint_6_angle | 第 3 踝关节角度 | true |
| 11 | joint_7_angle | 第 4 髋关节角度 | true |
| 12 | joint_8_angle | 第 4 踝关节角度 | true |
| 13 | body_x_velocity | 身体在世界 x 方向的前向速度 | true（前向速度奖励） |
| 14 | body_y_velocity | 身体在世界 y 方向的侧向速度 | true（侧滑惩罚） |
| 15 | body_z_velocity | 身体垂直速度 | true（跳跃/振颤抑制） |
| 16 | body_roll_velocity | 身体滚转角速度 | true（稳定性惩罚） |
| 17 | body_pitch_velocity | 身体俯仰角速度 | true |
| 18 | body_yaw_velocity | 身体偏航角速度 | true |
| 19 | joint_1_velocity | 第 1 髋关节角速度 | true |
| 20 | joint_2_velocity | 第 1 踝关节角速度 | true |
| 21 | joint_3_velocity | 第 2 髋关节角速度 | true |
| 22 | joint_4_velocity | 第 2 踝关节角速度 | true |
| 23 | joint_5_velocity | 第 3 髋关节角速度 | true |
| 24 | joint_6_velocity | 第 3 踝关节角速度 | true |
| 25 | joint_7_velocity | 第 4 髋关节角速度 | true |
| 26 | joint_8_velocity | 第 4 踝关节角速度 | true |

**重要**：环境未提供绝对世界坐标 (`x_position`, `y_position`) 和接触力信息。所有可用于奖励的信号均来自上述 27D 向量，`info` 字典为空且禁止使用任何内部拆解的奖励项。

## 4. 动作空间 action_space
- **type**: `Box` (连续)  
- **shape**: `[8]`  
- **dtype**: 通常为 `float32`，范围 `[-1.0, 1.0]` per joint  
- 每个动作维度对应一个关节的目标扭矩：

| 动作维度 | 名称 | 含义 |
|---|---|---|
| 0 | hip_1_torque | 第 1 髋关节扭矩 |
| 1 | ankle_1_torque | 第 1 踝关节扭矩 |
| 2 | hip_2_torque | 第 2 髋关节扭矩 |
| 3 | ankle_2_torque | 第 2 踝关节扭矩 |
| 4 | hip_3_torque | 第 3 髋关节扭矩 |
| 5 | ankle_3_torque | 第 3 踝关节扭矩 |
| 6 | hip_4_torque | 第 4 髋关节扭矩 |
| 7 | ankle_4_torque | 第 4 踝关节扭矩 |

扭矩范围归一化到 [-1, 1]，每个关节的加速度/力由底层模拟器转换成实际力矩。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无明确成功终止条件。没有目标到达或存活时长判定。
- **failure-like termination**: 
  - `body_height_outside_healthy_range`: 身体高度 `body_z` ≤ 0.2（崩溃/倒地）或 ≥ 1.0（过度跳跃/弹射）时环境结束，属于失败。
  - `state_value_outside_finite_range`: 任何状态值变为 NaN 或 Inf，属于失败。
- **ambiguous termination**: 无。
- **truncation**: `time_limit_reached` 触发截断，不表示成功或失败。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: **false**  
  观察信息中不存在任何“成功”标志。
- `explicit_failure_flag_available`: **false**  
  无法在奖励函数内部直接获取 `terminated` 或 `truncated`，只能在回合结束后从外部日志推断。任何 `info` 字段均不可用。
- `allowed_info_fields`: **无**（`info = {}`，严禁使用）
- `forbidden_or_uncertain_info_fields`: 所有 `info` 字段均禁止使用，尤其禁止以下已知但被隐藏的字段：
  - `reward_forward`, `reward_ctrl`, `reward_contact`, `reward_survive`
  - `x_position`, `y_position`, `distance_from_origin`

**结论**：奖励函数 **不能** 依赖成功/失败标志来施加终止奖励或分段奖励，所有引导必须来自于观测值（`obs`）和动作（`action`）本身。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

- **允许使用**：
  - `obs`：27D 当前观测
  - `action`：8D 动作
  - `next_obs`：27D 下一步观测
  - `info` 中 **明确允许的字段**（当前无任何允许字段，故 `info` 不可用）
  - `training_progress` **仅在 prompt 明确允许时才可使用**，当前未提及，一概禁止使用。

- **禁止使用**：
  - `original_reward`（官方奖励被遮盖）
  - `official_reward` 或任何从 `info` 中拼凑出的隐含奖励项
  - 未明确声明的 `info` 字段（如 `x_position`, `y_position`）
  - 未声明的 `obs` 切片（下标 0-26 均已在观察空间中声明，不得使用超出范围的索引）
  - 依靠全局状态或跨 Episode 累积量（函数必须是无状态的）

## 7. 可用于奖励函数的信号
- **位置/姿态**：`body_z`；四元数 `quat_w, quat_x, quat_y, quat_z` → 可计算 `body_up_z = 1 - 2*(quat_x² + quat_y²)`；8 个关节角度。
- **速度**：`body_x_velocity`、`body_y_velocity`、`



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
- 本文件只提供通用公式算子和少量动力学类型示例，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive: `w * signal`
  - penalty: `-w * abs(error)` 或 `-w * error**2`
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：无界状态值可能支配总奖励；状态值可能被占据/刷分，而不代表任务完成。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - `x / (1 + abs(x))`
  - `1 / (1 + k * abs(error))`
  - `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或 velocity/proximity 类信号容易被刷。
- 风险：threshold 过小会导致反馈饱和或无梯度。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。

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

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * (1 / (1 + k * abs(posture_error)))`
- 使用条件：前进/接近奖励导致不健康冲刺、翻倒或失稳。
- 风险：gate 太严格会抑制探索；跳跃类任务尤其要轻。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 风险：乘积容易塌缩；单一接触或单一事件不能当成功。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。





# Fresh Restart Evidence

- target_score: 2000.000
- best_score_so_far: 434.480

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| forward_progress + height_penalty + lateral_drift_penalty + upright_stability | 4 | 434.480 | 86.940 | unsolved |

## Previous interventions

- iter 2 (score=-224.850, structure=forward_progress + height_penalty + lateral_drift_penalty + upright_stability): 4. **selected_level**：Level 1——尺度修复。forward_progress 的系数过小导致其被 upright 信号完全主导，只需增大其系数即可改变相对重要性。 | 5. **selected_intervention**：仅将 forward_progress 的倍率从 2.0 改为 20.0，其他所有组件、结构、约束保持不变。
- iter 3 (score=265.020, structure=forward_progress + height_penalty + lateral_drift_penalty + upright_stability): 4. **selected_level**：Level 1，因forward_progress数学形态合理但相对upright_stability的尺度不足，只需调整其系数，不改变结构。 | 5. **selected_intervention**：将forward_progress的缩放系数从20.0提高至60.0，其他所有组件保持不变。
- iter 4 (score=86.940, structure=forward_progress + height_penalty + lateral_drift_penalty + upright_stability): 4. selected_level：Level 1——高度惩罚职责合理但系数偏小（episode_sum_mean 仅 -37.7，active_rate 5.5%），仅需加大惩罚幅度即可预期改善提前终止。 | 5. selected_intervention：只修改 height_penalty 组件，将边界外惩罚系数从 20.0 提升至 60.0。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.

```
