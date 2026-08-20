# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个平面内的单腿刚体机器人反复向前跳跃，保持躯干直立，同时合理使用关节力矩。核心目标是实现持续的向前移动（前进速度），而不是仅仅保持平衡。如果身体的任何状态超出健康范围（如躯干高度、倾角或状态值异常），回合将终止。截断条件为达到时间上限。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: (11,)
- dtype: 连续浮点数（由环境决定，通常为 float32 或 float64）
- obs[0]: 躯干高度（torso_height），垂直位置
- obs[1]: 躯干倾角（torso_angle），主身体的朝向角度
- obs[2]: 上腿关节角度（upper_joint_angle）
- obs[3]: 下腿关节角度（lower_joint_angle）
- obs[4]: 脚关节角度（foot_joint_angle）
- obs[5]: 前进速度（forward_velocity），身体的水平速度
- obs[6]: 垂直速度（vertical_velocity），身体的垂直速度
- obs[7]: 躯干角速度（torso_angular_velocity）
- obs[8]: 上腿关节角速度（upper_joint_speed）
- obs[9]: 下腿关节角速度（lower_joint_speed）
- obs[10]: 脚关节角速度（foot_joint_speed）

## 4. 动作空间 action_space
- type: Box (连续)
- 动作维度: 3，每个动作范围 [-1.0, 1.0]
- action 0: 上腿关节扭矩（upper_joint_torque）
- action 1: 下腿关节扭矩（lower_joint_torque）
- action 2: 脚关节扭矩（foot_joint_torque）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明确的成功终止信号，该任务一般通过截断表示存活到最大步数。
- failure-like termination: 以下任一条件触发立即终止，且通常视为摔倒或失控：
  - 躯干高度超出健康范围
  - 躯干倾角超出健康范围
  - 状态值超出有限健康范围（如 NaN、无穷大）
- ambiguous termination: 无。终止均源于明确的不健康状态。
- truncation: 达到预设时间步上限，回合被截断。可能表示机器人存活了完整周期，但无法直接用于奖励设计（需要根据观测自行判断性能）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（allowed_info_fields 为空）
- forbidden_or_uncertain_info_fields:
  - reward_forward、reward_ctrl、reward_survive（官方奖励分量）
  - x_position、y_position、z_distance_from_origin（全局位置）
  所有这些字段均禁止在奖励函数中使用。info 实际返回的字典为空（{}）。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：上一时刻观测，shape=(11,)
- action：动作，shape=(3,)
- next_obs：当前时刻观测，shape=(11,)
- info 中明确允许的字段：无（仅一个空字典）
- training_progress：只有当 prompt 明确允许时才能使用；本环境描述未提及允许使用，因此默认禁止

禁止使用：
- original_reward（已被掩码）
- 任何官方奖励函数成分
- 未声明的 info 字段（如真实的 x、y 坐标、奖励分量等）
- 未声明的 obs 切片（obs 维度含义已明确声明，可据以使用，但不得臆测未声明的含义）

## 7. 可用于奖励函数的信号
下列信号均可从 obs、next_obs 或 action 中直接获取，且未被禁止：
- 位置相关：
  - 躯干垂直高度（obs[0]）
  - 躯干倾角（obs[1]）
- 速度相关：
  - 前进速度（obs[5]），水平速度，是核心前进指标
  - 垂直速度（obs[6]）
  - 躯干角速度（obs[7]）
- 关节状态：
  - 关节角度（obs[2], obs[3], obs[4]）
  - 关节角速度（obs[8], obs[9], obs[10]）
- 动作/能耗：
  - 施加的关节扭矩（action[0], action[1], action[2]），可用于惩罚过大动作
- 接触信号：未提供

## 8. 不确定或不可用的信号
- 全局前进距离（x 坐标）：被禁止，且未在观测中暴露
- 官方奖励的任何分量：被禁止
- 明确的成功/失败标志：不存在
- 接触力或触地信号：观测中未包含
- 任意 info 字段：无效，因为 info 被清空



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