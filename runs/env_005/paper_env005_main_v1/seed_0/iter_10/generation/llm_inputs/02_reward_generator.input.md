# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个 3D 四足机器人（8 个扭矩驱动关节）向前行走或奔跑，同时保持身体直立。机器人的身体高度必须维持在一个健康的范围内（0.2 ~ 1.0），超出范围即终止。核心目标是维持稳定的前进运动，而不是仅仅保持平衡不动。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box（连续）
- shape: [105]
- dtype: 推测为 float32（连续物理量）
- obs[0]: 主躯干的垂直位置（body_z）
- obs[1]: 机体姿态四元数分量 1
- obs[2]: 机体姿态四元数分量 2
- obs[3]: 机体姿态四元数分量 3
- obs[4]: 机体姿态四元数分量 4
- obs[5]: 第 1 个关节角（hip_1）
- obs[6]: 第 2 个关节角（ankle_1）
- obs[7]: 第 3 个关节角（hip_2）
- obs[8]: 第 4 个关节角（ankle_2）
- obs[9]: 第 5 个关节角（hip_3）
- obs[10]: 第 6 个关节角（ankle_3）
- obs[11]: 第 7 个关节角（hip_4）
- obs[12]: 第 8 个关节角（ankle_4）
- obs[13]: 主躯干世界坐标系 x 方向速度（前进速度）
- obs[14]: 主躯干世界坐标系 y 方向速度（侧向速度）
- obs[15]: 主躯干世界坐标系 z 方向速度（垂直速度）
- obs[16]: 机体滚转角速度（roll）
- obs[17]: 机体俯仰角速度（pitch）
- obs[18]: 机体偏航角速度（yaw）
- obs[19]: 第 1 个关节角速度
- obs[20]: 第 2 个关节角速度
- obs[21]: 第 3 个关节角速度
- obs[22]: 第 4 个关节角速度
- obs[23]: 第 5 个关节角速度
- obs[24]: 第 6 个关节角速度
- obs[25]: 第 7 个关节角速度
- obs[26]: 第 8 个关节角速度
- obs[27:33]: 躯干（torso）接触力/力矩（fx, fy, fz, tx, ty, tz）
- obs[33:39]: 前左上腿接触力/力矩
- obs[39:45]: 前左下腿接触力/力矩
- obs[45:51]: 前左脚掌接触力/力矩（地面接触信号强）
- obs[51:57]: 前右上腿接触力/力矩
- obs[57:63]: 前右下腿接触力/力矩
- obs[63:69]: 前右脚掌接触力/力矩
- obs[69:75]: 后左上腿接触力/力矩
- obs[75:81]: 后左下腿接触力/力矩
- obs[81:87]: 后左脚掌接触力/力矩
- obs[87:93]: 后右上腿接触力/力矩
- obs[93:99]: 后右下腿接触力/力矩
- obs[99:105]: 后右脚掌接触力/力矩

## 4. 动作空间 action_space
- type: Box，连续，形状 [8]，每个维度范围 [-1.0, 1.0]
- action 0: 第 1 个髋关节扭矩（hip_1）
- action 1: 第 1 个踝关节扭矩（ankle_1）
- action 2: 第 2 个髋关节扭矩（hip_2）
- action 3: 第 2 个踝关节扭矩（ankle_2）
- action 4: 第 3 个髋关节扭矩（hip_3）
- action 5: 第 3 个踝关节扭矩（ankle_3）
- action 6: 第 4 个髋关节扭矩（hip_4）
- action 7: 第 4 个踝关节扭矩（ankle_4）

所有动作直接输出为关节扭矩，无需缩放即可施加到仿真器中。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功信号；可以认为 episode 因时间限制而截断（truncated）且未发生失败终止即为成功，即机器人存活并保持运动直到时间结束。
- failure-like termination: 身体高度超出健康范围（z < 0.2 或 z > 1.0）或状态中出现 NaN/inf 值。
- ambiguous termination: 无。
- truncation: 达到预设的时间步上限。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {}）
- forbidden_or_uncertain_info_fields: 全部 info 字段均不可用，包括任何可能的官方奖励项坐标等（已强制屏蔽）。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测）
- action（当前动作）
- next_obs（下一时刻观测）
- info 中明确允许的字段（此处为空，不可用）
- training_progress（仅当接口说明明确允许时可用，本环境无此许可，不可用）

禁止使用：
- original_reward（官方奖励已遮掩）
- 任何未声明的 info 字段（此处均未声明）
- 任何未在上述索引中说明的 obs 切片

## 7. 可用于奖励函数的信号
- position: 躯干垂直位置 (obs[0])，用于鼓励保持在健康范围中位。
- velocity: 前进速度 (obs[13] 的 body_x_velocity)，侧向速度 (obs[14])，垂直速度 (obs[15])，角速度 (obs[16:19])，各关节速度 (obs[19:27])。
- orientation: 躯干姿态四元数 (obs[1:5])，可判断身体倾斜程度。
- contact: 脚掌接触力 (obs[45:51] 前左脚、obs[63:69] 前右脚、obs[81:87] 后左脚、obs[99:105] 后右脚)，可用于判断支撑相与摆动相，鼓励合理步态。
- action/engine: 动作本身（action）可用于动作平滑性或能耗调节；关节位置 (obs[5:13]) 可用于观察姿态。

## 8. 不确定或不可用的信号
- 世界坐标 x/y 位置：未提供。
- 速度目标或期望前进速度：未提供。
- 能量/功耗直接测量：未提供（但可通过关节扭矩与速度间接估算，能耗本身不是硬指标）。
- 机器人实际世界朝向的绝对偏航：可以从四元数和关节状态推断，但 forward 方向是固定的世界 x 轴（obs[13] 直接给）。



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
| contact_reward + forward_encouragement + height_management + lateral_penalty + stability_cost | 1 | -19.200 | -19.200 | unsolved |
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
