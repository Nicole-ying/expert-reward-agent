# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一个连续控制的四足机器人行走任务。机器人拥有 8 个扭矩控制的关节（4 条腿，每条腿包含髋关节和踝关节），必须保持稳定站立并向前行进。本体高度必须维持在 0.2~1.0 的健康范围内，超出该范围（跌倒或弹飞）会立即终止回合。策略的核心目标是实现持续、稳定的前向运动，而非仅保持直立不动。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: (105,)
- dtype: 假定为 float32（连续值）
- obs[0]:  本体 z 坐标（垂直高度）
- obs[1]:  本体方向四元数分量 1
- obs[2]:  本体方向四元数分量 2
- obs[3]:  本体方向四元数分量 3
- obs[4]:  本体方向四元数分量 4
- obs[5]:  关节 1 角度（第1髋关节）
- obs[6]:  关节 2 角度（第1踝关节）
- obs[7]:  关节 3 角度（第2髋关节）
- obs[8]:  关节 4 角度（第2踝关节）
- obs[9]:  关节 5 角度（第3髋关节）
- obs[10]: 关节 6 角度（第3踝关节）
- obs[11]: 关节 7 角度（第4髋关节）
- obs[12]: 关节 8 角度（第4踝关节）
- obs[13]: 本体前进速度（世界坐标系 x 方向线速度）
- obs[14]: 本体侧向速度（世界坐标系 y 方向线速度）
- obs[15]: 本体垂直速度
- obs[16]: 本体滚转角速度
- obs[17]: 本体俯仰角速度
- obs[18]: 本体偏航角速度
- obs[19]: 关节 1 角速度
- obs[20]: 关节 2 角速度
- obs[21]: 关节 3 角速度
- obs[22]: 关节 4 角速度
- obs[23]: 关节 5 角速度
- obs[24]: 关节 6 角速度
- obs[25]: 关节 7 角速度
- obs[26]: 关节 8 角速度
- obs[27:33]:   躯干接触力/力矩（6D：[Fx, Fy, Fz, Tx, Ty, Tz]，世界坐标系）
- obs[33:39]:   前左大腿接触力/力矩
- obs[39:45]:   前左小腿接触力/力矩
- obs[45:51]:   前左脚接触力/力矩
- obs[51:57]:   前右大腿接触力/力矩
- obs[57:63]:   前右小腿接触力/力矩
- obs[63:69]:   前右脚接触力/力矩
- obs[69:75]:   后左大腿接触力/力矩
- obs[75:81]:   后左小腿接触力/力矩
- obs[81:87]:   后左脚接触力/力矩
- obs[87:93]:   后右大腿接触力/力矩
- obs[93:99]:   后右小腿接触力/力矩
- obs[99:105]:  后右脚接触力/力矩

说明：接触力块共 78 维，对应 13 个身体部位（躯干 + 四条腿的上段/下段/脚，共 1 + 4×3 = 13），每个部位 6 维力与力矩。

## 4. 动作空间 action_space
- type: Box（连续动作）
- shape: (8,)
- 范围：每个动作为 [-1.0, 1.0] 的扭矩值
- action 0: hip_1_torque  （第1条腿髋关节扭矩）
- action 1: ankle_1_torque（第1条腿踝关节扭矩）
- action 2: hip_2_torque   （第2条腿髋关节扭矩）
- action 3: ankle_2_torque（第2条腿踝关节扭矩）
- action 4: hip_3_torque   （第3条腿髋关节扭矩）
- action 5: ankle_3_torque（第3条腿踝关节扭矩）
- action 6: hip_4_torque   （第4条腿髋关节扭矩）
- action 7: ankle_4_torque（第4条腿踝关节扭矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明确成功信号。任务期望通过稳定行走最大化存活步数，并非到达某个目标点。
- failure-like termination: 身体高度超出 (0.2, 1.0) 范围（摔倒或弹飞）；状态值出现 NaN 或 Inf。
- ambiguous termination: 无（上述均属于失败终止）。
- truncation: 达到预设的最大步数时截断（时间限制）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（无任何允许的 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均被禁止使用，包括 reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin, x_velocity, y_velocity 等。info 在 step 中返回空字典，禁止依赖任何 info 内容。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前步的完整观察向量（105维）
- action：当前步采用的动作向量（8维）
- next_obs：执行动作后的下一观察向量（105维）
- info 中明确允许的字段：无（info 为空字典）
- training_progress：仅当 prompt 明确允许时可用，当前环境未授权使用

禁止使用：
- original_reward（已被掩码，不可用）
- official_reward 或任何官方奖励项
- 未声明的 info 字段（所有 info 字段均被禁止）
- 未声明的 obs 切片假设

## 7. 可用于奖励函数的信号
- position: 本体高度 obs[0]（用于保持健康范围）；身体垂直位移可间接反映跌倒倾向。
- velocity: 本体前进速度 obs[13]（主要推进目标）；侧向速度 obs[14]（希望抑制）；垂直速度 obs[15]（希望抑制过大波动）；本体角速度 obs[16-18] 可用来惩罚翻滚或俯仰。
- orientation: 四元数 obs[1-4] 可用于保持身体水平（例如惩罚偏离直立）。
- contact: 四条腿的脚部接触力（obs[45:51], obs[63:69], obs[81:87], obs[99:105]）可用于检测足部触地情况，鼓励对称或规避空踩。
- action/engine: 动作向量可用来施加能耗惩罚（如 -‖action‖²），鼓励平滑、低扭矩控制。

## 8. 不确定或不可用的信号
- info 中的所有字段（官方奖励项、位置信息等）均不可用。environment 明确禁止它们用于奖励设计。
- 无显式成功/失败标志，不能使用基于 info 的终止原因。
- 无世界坐标 x 绝对位置（但前进速度可直接观测，可通过累积速度增量间接反映“走得远”，但注意奖励函数是单步、无状态限制，不能累积跨步量，不过仍可基于当前速度进行奖励）。



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
- best_score_so_far: 12.310

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| angular_velocity_penalty + forward_velocity_reward + height_deviation_penalty + lateral_velocity_penalty + orientation_reward | 1 | 12.310 | 12.310 | unsolved |
| angular_velocity_penalty + forward_velocity_reward + lateral_velocity_penalty + orientation_reward | 1 | 11.050 | 11.050 | unsolved |
| forward_velocity_reward + orientation_reward + stability_penalty | 3 | -49.530 | -49.530 | unsolved |
| forward_velocity_reward + stability_penalty | 1 | -123.550 | -123.550 | unsolved |
| angular_velocity_penalty + forward_velocity_reward + height_boundary_penalty + lateral_velocity_penalty + orientation_reward | 1 | -131.920 | -131.920 | unsolved |

## Previous interventions

- iter 2 (score=-50.900, structure=forward_velocity_reward + orientation_reward + stability_penalty): selected_level**: Level 2 — 必要信号缺失（身体倾斜反馈不可达），仅调系数无法解决17/20崩塌问题；干预类型为添加连续姿态奖励组件（sparse_to_dense变体：将隐式高度终止风险转化为显式倾斜反馈）。 | selected_intervention**: 新增`orientation_reward`组件，利用body quaternion计算身体z轴在世界坐标系中的投影`body_up_z`，奖励保持直立姿态。
- iter 3 (score=-137.450, structure=forward_velocity_reward + orientation_reward + stability_penalty): `selected_level`: Level 1 — the stability_penalty component has correct structure (hinge loss with appropriate margins for the healthy range), but its height sub-coefficient w_h=10 is too small relative to forward_veloci | `selected_intervention`: increase w_h from 10.0 to 50.0, a 5× amplification of the height penalty within the stability_penalty component. No other coefficients, margins, or mathematical forms are changed.
- iter 4 (score=-49.530, structure=forward_velocity_reward + orientation_reward + stability_penalty): `selected_level`: Level 2 — `global_to_local_gate`。触发条件：(1) forward_velocity_reward 是唯一主信号但其不顾姿态始终全额发放，证据为18/20失败终止时 forward 仍贡献42.5% signed_share；(2) 纯系数调整(w_h 10→50)已证实无效且恶化；(3) 需要结构变换将前进奖励门控在稳定状态上。 | `selected_intervention`: 唯一修改 forward_velocity_reward 组件，将其从 `w_fwd * body_x_vel` 变换为 `w_fwd * body_x_vel * stability_gate`，其中 stability_gate 由身体直立度与高度健康度连续乘积构成；其他组件保持 best 代码不变（w_h=10.0）。
- iter 6 (score=12.310, structure=angular_velocity_penalty + forward_velocity_reward + height_deviation_penalty + lateral_velocity_penalty + orientation_reward): 4. `selected_level`: Level 2** — a necessary signal (height management) is entirely absent, and the termination evidence (19/20 height-violation failures) directly supports adding it. This is the `unbounded_to_bounded` p | 5. `selected_intervention`

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
