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
