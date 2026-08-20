# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个平面单腿铰接体，使其在保持直立和身体状态健康的前提下，持续向前跳跃前进。主要目标是产生稳定的前向位移，同时避免摔倒或进入无效状态（高度过低/过高、躯干角度过大、状态值非有限）。策略应追求持续的前向移动，而不仅仅是不摔倒。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box (连续向量)
- shape: (11,)
- dtype: float32 或 float64（连续值）
- obs[0]: torso_height —— 躯干（主体）的垂直位置
- obs[1]: torso_angle —— 躯干的朝向角度
- obs[2]: upper_joint_angle —— 大腿关节角度
- obs[3]: lower_joint_angle —— 小腿关节角度
- obs[4]: foot_joint_angle —— 足部关节角度
- obs[5]: forward_velocity —— 躯干水平方向速度（前向速度）
- obs[6]: vertical_velocity —— 躯干垂直方向速度
- obs[7]: torso_angular_velocity —— 躯干角速度
- obs[8]: upper_joint_speed —— 大腿关节角速度
- obs[9]: lower_joint_speed —— 小腿关节角速度
- obs[10]: foot_joint_speed —— 足部关节角速度

## 4. 动作空间 action_space
- type: Box (连续向量)
- shape: (3,)
- 范围: 每个动作维度 [-1.0, 1.0]
- action 0: upper_joint_torque —— 施加在大腿关节上的扭矩
- action 1: lower_joint_torque —— 施加在小腿关节上的扭矩
- action 2: foot_joint_torque —— 施加在足部关节上的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功信号；任务没有“目标到达”的终点，前向移动是可无限持续的，因此不存在成功终止。
- failure-like termination:
  - body_height_outside_healthy_range （躯干高度超出健康范围）
  - torso_angle_outside_healthy_range （躯干角度超出健康范围）
  - state_value_outside_finite_healthy_range （状态值中出现非有限值，如NaN或无穷大）
  以上三种条件触发 `terminated=True`，均视为失败。
- ambiguous termination: 无。
- truncation: 达到 `time_limit_reached` 时触发 `truncated=True`，不算失败，仅截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false（只能通过 `terminated` 标志间接推断，但 `info` 为空字典，不包含失败原因字段）
- allowed_info_fields: [] （无可用 info 字段）
- forbidden_or_uncertain_info_fields:
  - reward_forward
  - reward_ctrl
  - reward_survive
  - x_position
  - y_position
  - z_distance_from_origin
  等官方奖励项和世界位置信息均被禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前观察向量（形状 (11,)）
- action: 当前执行的动作向量（形状 (3,)）
- next_obs: 下一时刻观察向量（形状 (11,)）
- info: 允许使用的字段为空（即 `info` 不可用）
- training_progress: 只有在 prompt 明确说明时才使用（本次说明未提及，故不应依赖）

禁止使用：
- original_reward（官方奖励已被屏蔽）
- official_reward
- 任何未声明的 info 字段（包括那些被禁止的字段）
- 未在观察空间中明确定义的 obs 切片（例如不能直接访问世界坐标 x 位置，因为不在观察中）

## 7. 可用于奖励函数的信号
- position:
  - torso_height (obs[0])：用于判断重心是否在健康范围，可构造生存/高度奖励
  - torso_angle (obs[1])：用于鼓励直立
  - 关节角度 obs[2], obs[3], obs[4]：可构造姿态或效率项
- velocity:
  - forward_velocity (obs[5])：核心信号，直接衡量前向移动速度，可用于正奖励
  - vertical_velocity (obs[6])：可避免过大的上下跳动
  - 关节速度 obs[8], obs[9], obs[10]：可构造平滑性或能耗惩罚
- orientation:
  - torso_angle, torso_angular_velocity (obs[1], obs[7])：保持直立
- contact: 无直接接触信号，但可通过高度、速度间接推断。
- action/engine:
  - 动作向量 (action[0], action[1], action[2])：可惩罚大扭矩动作（例如平方和）

## 8. 不确定或不可用的信号
- 世界坐标 x（前向位移）：被禁止，不可用。
- 官方奖励分量（reward_forward, reward_ctrl, reward_survive）：被屏蔽，不可用。
- 接触力、接地状态：未提供。
- info 字典中任何字段：均为空，不可用。
- 历史累积信息（因 reward 函数无状态，不能跨步累积速度或距离）。



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

- target_score: 3800.000
- best_score_so_far: 2926.190

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| ang_vel_penalty + speed_posture_reward | 3 | 2926.190 | 2509.870 | unsolved |
| ang_vel_penalty + speed_posture_reward + vertical_reward | 1 | 2080.000 | 2080.000 | unsolved |
| ang_vel_penalty + energy_penalty + speed_posture_reward | 1 | 724.240 | 724.240 | unsolved |

## Previous interventions

- iter 2 (score=2606.950, structure=ang_vel_penalty + speed_posture_reward): `selected_level`：Level 1（尺度修复）——posture_factor的指数衰减形态和符号均正确，但k_angle=10.0对需要自然躯干振荡的跳跃步态过于严厉，使有效奖励在中高速时被过度压缩。 | `selected_intervention`：单一组件修改——将posture_factor中的k_angle从10.0降至5.0，其余所有组件和系数不变。
- iter 3 (score=2509.870, structure=ang_vel_penalty + speed_posture_reward): `selected_level`: Level 2 — 同一骨架家族已迭代2轮未刷新best，且第2轮尺度修复（k_angle 10→5）未改善反而恶化，触发从Level 1升级到结构变换；证据模式为乘积耦合可能限制任务所需的行为空间。 | `selected_intervention`: 将`speed_posture_reward`从乘法耦合`forward_vel * exp(-k*angle²)`改为加法解耦`forward_vel - w_posture * angle²`，w_posture=5.0（使0.2rad倾角时惩罚约0.2，约为典型forward_vel的13%），ang_vel_penalty保持不变。
- iter 4 (score=724.240, structure=ang_vel_penalty + energy_penalty + speed_posture_reward): `selected_level`: **Level 2**。触发条件：必要职责缺失（动作效率约束），且Level 1仅调系数已在iter1→iter2→iter3中反复尝试未刷新best。 | `selected_intervention`: 在iter1乘法骨架基础上**新增energy_penalty组件**（对三关节扭矩的平方和施加轻量惩罚），不改变speed_posture_reward和ang_vel_penalty的现有数学形态。
- iter 5 (score=2080.000, structure=ang_vel_penalty + speed_posture_reward + vertical_reward): `selected_level`: Level 2 — 需要加入新的跳跃引导信号来缩小2926→3800的差距，不能简单回退best。 | `selected_intervention`: 以best代码为基础，新增`vertical_reward`组件：用`max(0, next_obs[6])`奖励向上的垂直速度，系数0.3，鼓励更有力的跳跃以提升前进效率。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
