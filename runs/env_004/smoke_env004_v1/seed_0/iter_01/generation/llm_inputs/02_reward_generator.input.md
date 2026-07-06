# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个平面单腿关节体持续向前跳跃，同时保持躯干直立且各关节处于健康范围内。摔倒、躯干倾角过大或任何身体状态超出安全范围将立即终止回合。策略的核心是产生持续的前向位移，而非仅仅保持平衡。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box（连续）
- shape: (11,)
- dtype: 连续浮点数（具体精度由环境决定，通常为 float32）
- obs[0]: torso_height – 躯干（主体）的垂直高度
- obs[1]: torso_angle – 躯干的倾斜角度（方向角）
- obs[2]: upper_joint_angle – 上腿关节角度
- obs[3]: lower_joint_angle – 下腿关节角度
- obs[4]: foot_joint_angle – 足部关节角度
- obs[5]: forward_velocity – 躯干的水平前向速度
- obs[6]: vertical_velocity – 躯干的垂直速度
- obs[7]: torso_angular_velocity – 躯干的角速度
- obs[8]: upper_joint_speed – 上腿关节角速度
- obs[9]: lower_joint_speed – 下腿关节角速度
- obs[10]: foot_joint_speed – 足部关节角速度

## 4. 动作空间 action_space
- type: Box（连续）
- shape: (3,)
- 每个维度取值范围：[-1.0, 1.0]
- action 0: upper_joint_torque – 施加在上腿铰链的力矩
- action 1: lower_joint_torque – 施加在下腿铰链的力矩
- action 2: foot_joint_torque – 施加在足部铰链的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明确定义的成功终止条件。任务目标是持续前进，没有预设的成功状态（如到达某个位置）。
- failure-like termination:
  - body_height_outside_healthy_range：躯干高度超出健康范围（典型为摔倒在地面或跳跃过高）
  - torso_angle_outside_healthy_range：躯干倾角过大（身体过度倾斜）
  - state_value_outside_finite_healthy_range：任何状态值变为无限/NaN（数值不稳定，通常由关节极限或碰撞导致）
- ambiguous termination: 无
- truncation: time_limit_reached（达到最大步数限制），为环境截断，不代表成功或失败，仅表示时间耗尽。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，包括 reward_forward, reward_ctrl, reward_survive, x_position, y_position, z_distance_from_origin 等；这些字段被明确禁止，且在实际环境中已被屏蔽。terminated/truncated 标志不通过 info 提供，也无法在 reward 函数中直接使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs：形状 (11,) 的 numpy 数组，当前步的观察
- action：形状 (3,) 的 numpy 数组，当前步执行的动作
- next_obs：形状 (11,) 的 numpy 数组，执行动作后的观察
- info：固定为空 dict {}，不可从中读取任何字段
- training_progress：仅当 prompt 明确允许时才使用，否则忽略

禁止使用：
- original_reward：已屏蔽，不得依赖
- official_reward：等价于 original_reward，不可用
- info 中的所有字段：当前 info 为空，任何尝试读取都会失败
- 未在观察空间中声明的任何数据

## 7. 可用于奖励函数的信号
- position: 躯干高度（obs[0]），躯干倾角（obs[1]），各关节角度（obs[2‑4]）
- velocity: 前向速度（obs[5]），垂直速度（obs[6]），躯干角速度（obs[7]），关节角速度（obs[8‑10]）
- orientation: 躯干倾角可直接使用
- contact: 无直接接触传感器，但躯干高度过低可间接反映脚着地或摔倒
- action/engine: 动作力矩值（动作向量），可用于调节幅度或平滑性约束

## 8. 不确定或不可用的信号
- 真实的全局前向位移（x_position）被隐藏，不可用
- 任何形式的官方奖励分量（reward_forward, reward_ctrl, reward_survive 等）被屏蔽
- 健康状态的二元标志（身体是否超出范围）无法直接获得，只能从观察值推测
- 终止信号（terminated）不传入 reward 函数，不可用于奖励计算



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