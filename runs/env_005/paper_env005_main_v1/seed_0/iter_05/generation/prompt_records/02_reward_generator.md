# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境背景；
2. expert_reward_context.md：RAG 检索并压缩后的专家知识；
3. optional masked_step_source：默认不提供，除非调试开启。

你的任务：
直接生成第一版奖励函数 `reward_v1.py`，并附带一份简短设计说明。

# 总体设计原则

- 从简单到复杂，但“简单”不等于只有一个组件。
- 不要用“最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 不要机械照抄 route 推荐公式。
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

以下原则适用于所有环境和任务类型。expert_reward_context.md 中的骨架是奖励设计原语、风险提示和参考起点，不构成封闭候选集合。你可以直接采用、组合、变形这些原语，也可以根据任务目标和可用环境信号提出新的数学结构。

## 原则 1：信号可用性优先

- 先检查 environment_card.md 和 expert_reward_context.md 中声明的可用信号（obs indices、info 字段、termination 信息）。
- 只有当信号确实存在于环境接口中时，才设计依赖该信号的组件。
- 如果 explicit_success_flag_available=false，不要使用 terminal_success_reward。
- 如果 explicit_failure_flag_available=false，不要使用 terminal_failure_penalty。
- 不要发明未声明的 info 字段或 obs 切片。

## 原则 2：稠密性

- 优先选择每步都能提供有意义梯度的连续信号。
- 二值条件信号（如只在特定区域触发的 bonus）触发率过低时等于摆设。
- 连续函数（线性、bounded、乘积）比离散条件更有利于学习。

## 原则 3：尺度与平衡

- 不同组件的量级应大致可比，不要让一个组件在数值上统治其他组件。
- 约束/惩罚不应在数值上无条件压制任务驱动力；具体尺度必须结合其触发频率、数学形态和预期行为判断，不能套用固定比例阈值。
- 差分信号、持续状态奖励和稀疏事件奖励具有不同的时间语义，不能仅凭步均值比例判断谁更重要。

## 原则 4：信号冲突

- 不要同时大权重使用两个计算同一物理量的信号（如 distance 和 progress_delta），它们会产生重复梯度。
- 不要让惩罚项压制探索——过于严厉的姿态/速度约束可能导致 agent 不敢行动。

## 原则 5：阶段条件

- v1 阶段避免过早引入效率/动作代价（energy_penalty、time_penalty），agent 应先学会完成任务再优化效率。
- 复杂门控（gated_reward）、动态课程（dynamic curriculum）默认后续迭代再加入。

## 原则 6：可利用风险

- 每个组件都要考虑 agent 可能找到的捷径。例如：只奖励接近目标可能导致在目标附近震荡而不完成；只奖励接触可能诱导 agent 反复轻触地面。
- 如果使用 contact 或类似离散事件，必须和其他连续条件组合，不能单独作为成功信号。

## 原则 7：最小设计

- reward_v1 推荐使用 2~4 个组件，每个必须有明确角色，不能为了显得完整而堆叠。
- 信号多的时候要主动取舍——不是所有可用信号都需要放进 v1。优先选择对任务目标最核心的信号。
- 把 expert_reward_context.md 中按任务路由推荐的骨架作为思考面，而不是允许列表。最终设计由任务目标、可用信号和预期策略行为决定，不要求组件必须对应某个 skeleton_id。
- components dict 里只放 total_reward 公式中等号右边的变量。如 `total_reward = A + B + C`，则 components 只有 A、B、C。不要把 `total_reward` 放进 components，也不要把 A 的计算中间变量（如 A = a1+a2+a3 中的 a1）放进 components。

# role-based component budget

以上原则的具体落地方式：v1 推荐使用 2~4 个组件，按以下角色组织。检索骨架只提供设计启发，不限制你组合、变形或创造适合当前环境的新信号。

## 必须包含

**1 个主学习信号。** 这是 reward 的核心驱动力，告诉 agent "做什么能得分"。主信号的特征：
- 每步都有梯度（稠密），不是只在终点或特定事件触发
- 与任务目标直接相关（靠近目标、前进、保持存活等）
- 在策略学习中承担主要任务驱动作用；它不一定是每一步数值最大的正向项
- components key 应准确描述其物理或任务含义，不强制命名为 `progress_reward`

主信号可以来自检索骨架，也可以是对多个设计原语的组合、变形或基于环境事实构造的新形式。不要为了匹配骨架名称而牺牲任务语义。

## 允许包含（按需，不是必须全加）

- **0~2 个稳定/安全/平滑约束。** 如果任务需要控制速度、姿态、动作抖动等，可以加入轻量惩罚项。约束的角色是"方向盘"而非"刹车"——权重应明显小于主信号。如果环境不需要（如离散动作空间不需要 action_smoothness），不加。
- **0~1 个任务完成近似信号（proxy）。** 如果环境没有显式 success flag 但需要在 agent 接近完成时给予额外引导，可以用多条件组合的 soft proxy。proxy 必须由多个连续条件组合（不是单一二值条件），且不能直接伪造 success flag。
- **0~1 个效率/动作代价。** 权重必须很小。v1 阶段 agent 应先学会完成任务，效率优化后续迭代再加入。

## 默认不在 v1 使用

- terminal_success_reward（需显式 success flag，且 flag 在 info 中实际可用）
- terminal_failure_penalty（需显式 failure flag 或明确的 termination_reason）
- gated_reward（多阶段门控，复杂且容易过严）
- dynamic_curriculum_reward（依赖训练进度，v1 无历史参考）

## 避免重复

- 不要同时大权重使用两个计算同一物理量的信号（如 distance_reward 和 progress_delta_reward）。如果两个都用了，一个必须是主信号，另一个只能是小权重辅助锚点——但更建议只选一个。

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

为了兼容旧 wrapper，也可以把 components 写入 `info["reward_terms"]`，但仍然建议返回 `(float(total_reward), components)`。

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
- components 只包含被加到 total_reward 的组件（A、B、C），**不包含 total_reward 本身**。

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
- 使用了哪些奖励组件；
- 每个组件的角色；
- 为什么没有使用 terminal_success_reward / terminal_failure_penalty；
- 哪些组件留到后续迭代；
- 训练后应该观察哪些 failure mode。

```

## User Prompt

```markdown
# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个连续控制运动任务。一个三维四足机器人具有四条腿和八个力矩控制关节，必须持续向前行走或奔跑，同时保持身体直立。机器人的身体高度必须维持在健康范围（0.2 < z < 1.0）之内；若身体高度低于 0.2（倒塌）或高于 1.0（弹跳出范围）或任何状态值变为非有限，则回合提前终止。策略应实现稳定且持续的前向运动，而非仅仅保持站立。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box（连续值）
- shape: (105,)
- dtype: float32（推断，无明确说明）
- obs[0]：主体垂直位置 body_z
- obs[1:5]：主体方向四元数（body_orientation_1 ~ body_orientation_4，4 个分量）
- obs[5:13]：8 个关节角度（joint_1_angle ~ joint_8_angle）
- obs[13]：世界坐标系前向速度 body_x_velocity
- obs[14]：世界坐标系横向速度 body_y_velocity
- obs[15]：垂直速度 body_z_velocity
- obs[16]：滚动角速度 body_roll_velocity
- obs[17]：俯仰角速度 body_pitch_velocity
- obs[18]：偏航角速度 body_yaw_velocity
- obs[19:27]：8 个关节角速度（joint_1_velocity ~ joint_8_velocity）
- obs[27:105]：外部接触力与力矩（78 维）。按 13 个身体部件（躯干 + 4 条腿的上段、下段、足段）各 6 维 [fx, fy, fz, tx, ty, tz] 排列。足部接触信号较强的索引：
  - obs[45:51]：前左脚（front_left foot）
  - obs[63:69]：前右脚（front_right foot）
  - obs[81:87]：后左脚（back_left foot）
  - obs[99:105]：后右脚（back_right foot）

## 4. 动作空间 action_space
- type: Box（连续，力矩归一化到 [-1, 1]）
- shape: (8,)
- 每个动作对应一个关节的力矩指令：
  - action 0: hip_1（前左髋关节力矩）
  - action 1: ankle_1（前左踝关节力矩）
  - action 2: hip_2（前右髋关节力矩）
  - action 3: ankle_2（前右踝关节力矩）
  - action 4: hip_3（后左髋关节力矩）
  - action 5: ankle_3（后左踝关节力矩）
  - action 6: hip_4（后右髋关节力矩）
  - action 7: ankle_4（后右踝关节力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无（不存在明确的成功触发条件）
- failure-like termination: 身体高度超出 (0.2, 1.0) 范围或任何状态值为 NaN/Inf
- ambiguous termination: 无
- truncation: 达到内部时间限制（time_limit_reached）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典，不可用）
- forbidden_or_uncertain_info_fields: 所有字段均被禁止或不存在，特别是 reward_forward、reward_ctrl、reward_contact、reward_survive、x_position、y_position、distance_from_origin、x_velocity、y_velocity 等均不可用。

*注：由于 info 为空，终止原因不能直接从标志位获取。然而，reward 函数可以通过 next_obs[0]（body_z）是否在 [0.2, 1.0] 之外来推断是否发生失败式终止；同样，若 next_obs 包含 NaN 或 Inf，则可能需要防御性处理。*

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前状态向量 (105,)
- action: 当前动作向量 (8,)
- next_obs: 下一状态向量 (105,)
- info 中明确允许的字段：无（所有字段被禁止）
- training_progress：仅当 prompt 明确允许时使用（本环境 prompt 未允许，故不应使用）

禁止使用：
- original_reward（已被遮蔽，不可用）
- official_reward 或其任何成分
- 未声明的 info 字段或 obs 切片
- 任何全局位置信息（x/y 坐标）或全局速度（x/y 速度）除非在 obs 中显式给出

## 7. 可用于奖励函数的信号
- position: body_z (obs[0])、主体姿态四元数 (obs[1:5])、关节角度 (obs[5:13])
- velocity: body_x_velocity (obs[13])、body_y_velocity (obs[14])、body_z_velocity (obs[15])、角速度 (obs[16:18])、关节角速度 (obs[19:27])
- orientation: 主体方向四元数（可转换为倾斜程度）
- contact: 各身体部件的接触力/力矩 (obs[27:105])，尤其是四个足部的接触信号
- action/engine: 动作本身 (action, 8维) 可用于惩罚过大力矩或抖动

## 8. 不确定或不可用的信号
- 官方奖励项（reward_forward、reward_ctrl 等）被强制屏蔽，不可使用
- 全局绝对位置（x_position, y_position, distance_from_origin）不在观察中且 info 禁止
- 速度字段 x_velocity、y_velocity 在 info 中禁止，但 body_x_velocity (obs[13]) 仍可用
- 明确的成功/失败标志不存在
- 剩余步数或时间信息不可直接获得
- training_progress 是否可用不确定（未明确说明，建议不使用）
- 终止原因标签（如 fell_over、height_fault 等）不存在



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架是任务相关的设计原语、风险提示和参考起点，不构成封闭候选集合。可直接采用、组合、变形，或基于环境事实提出新的数学结构。

## 1. 任务路由摘要
- locomotion_continuous_control：任务目标是稳定前进通过地形。重点观察 fast_then_fail / hover_or_stand_still / over_conservative_policy。

## 2. 相关奖励骨架摘要（按任务路由检索）

以下骨架由任务路由检索推荐。是否使用某个骨架取决于：
1. 该骨架所需信号是否在环境接口中实际可用；
2. 是否与该任务阶段匹配（v1 优先设计核心学习信号，效率/安全类后续迭代加入）。

### forward_progress_reward
- 角色: 前进方向引导
- 数学形态: lambda_fwd * forward_velocity
- 需要信号: forward velocity component
- 使用说明: 奖励沿前进方向的速度。适合 locomotion 类任务。
- 风险: 快速前进但容易摔倒。
- 后续迭代: 若 fast_then_fail，配合稳定性约束。

### terminal_failure_penalty
- 角色: 失败惩罚
- 数学形态: -R_failure * I[failure]
- 需要信号: 显式 failure flag 或 termination_reason
- 使用说明: 当环境提供显式 failure flag 时可用。若 explicit_failure_flag_available=false，不可使用。
- 风险: 误判终止原因。
- 后续迭代: 当能区分失败终止后再加。

### energy_penalty
- 角色: 动作/能耗约束
- 数学形态: -lambda_action * engine_use(action)
- 需要信号: action
- 使用说明: 惩罚动作幅度/能耗。先完成任务再加入，v1 太早加入可能不敢动。
- 风险: agent_afraid_to_move。
- 后续迭代: 能完成任务并稳定后再优化能耗。

### alive_bonus
- 角色: 存活激励
- 数学形态: lambda_alive * I[not_done]
- 需要信号: done flag
- 使用说明: 每步给予小额存活奖励，鼓励 agent 不提前终止。适合 locomotion/balance 类任务。
- 风险: hover_or_stand_still（原地不动来获取存活奖励）。
- 后续迭代: 若 agent 不动，减小权重或配合前向奖励。

### action_smoothness_penalty
- 角色: 动作平滑约束
- 数学形态: -lambda_smooth * |action - previous_action|
- 需要信号: previous action or action history
- 使用说明: 惩罚动作的剧烈变化。适合连续控制任务。
- 风险: 离散动作空间不可用（无历史信息）。
- 后续迭代: 若动作抖动，增大权重。

### stability_penalty
- 角色: 轻量稳定约束
- 数学形态: -lambda_v*|velocity| - lambda_a*|angle| - lambda_w*|angular_velocity|
- 需要信号: next_obs[2], next_obs[3], next_obs[4], next_obs[5]
- 使用说明: 抑制高速、大角度或高角速度。适合需要稳定运动或姿态控制的任务。
- 风险: 过强会保守或不敢动。
- 后续迭代: 若高速失稳，增大权重。

## 3. reward_v1 生成要求
- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget：每个组件必须有明确角色，不能为了显得完整而堆叠。
- 将上述骨架作为思考面而非允许列表；最终设计由任务目标、可用信号和预期策略行为决定，不要求组件对应已有 skeleton_id。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。



# Fresh Restart Evidence

- target_score: 6000.000
- best_score_so_far: -49.530

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| forward_velocity_reward + orientation_reward + stability_penalty | 3 | -49.530 | -49.530 | unsolved |
| forward_velocity_reward + stability_penalty | 1 | -123.550 | -123.550 | unsolved |

## Previous interventions

- iter 2 (score=-50.900, structure=forward_velocity_reward + orientation_reward + stability_penalty): selected_level**: Level 2 — 必要信号缺失（身体倾斜反馈不可达），仅调系数无法解决17/20崩塌问题；干预类型为添加连续姿态奖励组件（sparse_to_dense变体：将隐式高度终止风险转化为显式倾斜反馈）。 | selected_intervention**: 新增`orientation_reward`组件，利用body quaternion计算身体z轴在世界坐标系中的投影`body_up_z`，奖励保持直立姿态。
- iter 3 (score=-137.450, structure=forward_velocity_reward + orientation_reward + stability_penalty): `selected_level`: Level 1 — the stability_penalty component has correct structure (hinge loss with appropriate margins for the healthy range), but its height sub-coefficient w_h=10 is too small relative to forward_veloci | `selected_intervention`: increase w_h from 10.0 to 50.0, a 5× amplification of the height penalty within the stability_penalty component. No other coefficients, margins, or mathematical forms are changed.
- iter 4 (score=-49.530, structure=forward_velocity_reward + orientation_reward + stability_penalty): `selected_level`: Level 2 — `global_to_local_gate`。触发条件：(1) forward_velocity_reward 是唯一主信号但其不顾姿态始终全额发放，证据为18/20失败终止时 forward 仍贡献42.5% signed_share；(2) 纯系数调整(w_h 10→50)已证实无效且恶化；(3) 需要结构变换将前进奖励门控在稳定状态上。 | `selected_intervention`: 唯一修改 forward_velocity_reward 组件，将其从 `w_fwd * body_x_vel` 变换为 `w_fwd * body_x_vel * stability_gate`，其中 stability_gate 由身体直立度与高度健康度连续乘积构成；其他组件保持 best 代码不变（w_h=10.0）。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.

```
