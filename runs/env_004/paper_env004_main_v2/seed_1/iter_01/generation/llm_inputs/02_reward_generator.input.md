# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个平面单腿铰接体，通过三个关节的连续力矩驱动，使其持续向前跳跃移动。策略必须在保持身体直立、关节力矩高效的前提下，尽可能产生正向的水平位移。跌倒、身体高度异常、躯干倾角过大或状态数值溢出都会导致回合立即结束。奖励设计应鼓励“持续向前推进”，而非仅仅“保持不倒”。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: [11]
- dtype: float32 (假设，通常为float)
- obs[0]: torso_height —— 主体质心垂直高度
- obs[1]: torso_angle —— 躯干相对于垂直方向的倾角
- obs[2]: upper_joint_angle —— 大腿关节角度
- obs[3]: lower_joint_angle —— 小腿关节角度
- obs[4]: foot_joint_angle —— 足部关节角度
- obs[5]: forward_velocity —— 主体水平速度（向前为正）
- obs[6]: vertical_velocity —— 主体垂直速度
- obs[7]: torso_angular_velocity —— 躯干角速度
- obs[8]: upper_joint_speed —— 大腿关节角速度
- obs[9]: lower_joint_speed —— 小腿关节角速度
- obs[10]: foot_joint_speed —— 足部关节角速度

## 4. 动作空间 action_space
- type: Box，连续动作，范围 [-1.0, 1.0]
- action 0: upper_joint_torque —— 大腿关节力矩
- action 1: lower_joint_torque —— 小腿关节力矩
- action 2: foot_joint_torque —— 足部关节力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止条件；任务目标是持续前进，并不存在“到达终点”的概念，因此没有成功终止。
- failure-like termination: body_height_outside_healthy_range（身体高度超标，可能跌倒或跳太高）、torso_angle_outside_healthy_range（躯干倾角过大，即将倒下）、state_value_outside_finite_healthy_range（观测值出现 NaN 或 inf）均属于失败终止。
- ambiguous termination: 无。
- truncation: time_limit_reached，回合因达到最大步数而截断，既非成功也非失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: **空**（任何 info 字段均不得使用）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_survive, x_position, y_position, z_distance_from_origin 等（所有列出的字段均禁止使用，且 info 实际返回 {}）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前观测向量
- action：当前动作向量
- next_obs：下一时刻观测向量
- training_progress：**仅当 prompt 明确声明时**才允许使用（本环境未声明，应禁止使用）

禁止使用：
- original_reward（官方奖励已被屏蔽）
- info 中的任何字段（allowed_info_fields 为空，且所有有意义字段均在 forbidden_info_fields 中）
- 任何未在观测空间中声明的信号（如世界坐标 x、y、z_distance）
- 奖励函数必须是无状态的，不能跨 step 累加信息

## 7. 可用于奖励函数的信号
- position: torso_height（obs[0]）—— 可用于惩罚过高或过低，保持健康高度区间
- velocity: forward_velocity（obs[5]）、vertical_velocity（obs[6]）、torso_angular_velocity（obs[7]）、upper_joint_speed（obs[8]）、lower_joint_speed（obs[9]）、foot_joint_speed（obs[10]）
- orientation: torso_angle（obs[1]）—— 可惩罚偏离直立的角度
- contact: 无直接接触传感器（仅能通过高度和倾角间接推断脚与地面的接触情况）
- action/engine: action[0], action[1], action[2] —— 可惩罚过大的力矩或过快的动作变化

## 8. 不确定或不可用的信号
- 绝对世界位置（x, y, z_distance）不可用，禁止通过 info 或观测获取。
- 任何来自 info 的奖励分量（reward_forward, reward_ctrl, reward_survive 等）均不可用。
- 没有明确的 success/failure flag，无法通过标志位直接判断成功或失败。
- 无接触力、地面反作用力等物理交互信号。
- 回合持续时间步数不可直接用于奖励（函数无状态）；但可通过 training_progress 提供全局进度（本环境未允许使用）。



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