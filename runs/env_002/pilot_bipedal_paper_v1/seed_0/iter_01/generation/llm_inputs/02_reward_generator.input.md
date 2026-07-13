# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个两足身体在不平坦地形上尽可能远、尽可能快地向前行走，同时尽量减小能量消耗。智能体需要协调两条腿的髋关节和膝关节，产生稳定的步态。如果身体摔倒，回合立即结束；如果走到地形尽头，也会终止。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32
- obs[0]: hull_angle — 身体相对于竖直方向的倾角（弧度）
- obs[1]: hull_angular_velocity — 身体的角速度（弧度/秒）
- obs[2]: horizontal_velocity — 前向/后向的线速度（米/秒）
- obs[3]: vertical_velocity — 上下方向的线速度（米/秒）
- obs[4]: hip1_angle — 腿1的髋关节角度（弧度）
- obs[5]: hip1_speed — 腿1的髋关节角速度（弧度/秒）
- obs[6]: knee1_angle — 腿1的膝关节角度（弧度）
- obs[7]: knee1_speed — 腿1的膝关节角速度（弧度/秒）
- obs[8]: leg1_contact — 腿1与地面的接触标志（1.0 接触，0.0 未接触）
- obs[9]: hip2_angle — 腿2的髋关节角度（弧度）
- obs[10]: hip2_speed — 腿2的髋关节角速度（弧度/秒）
- obs[11]: knee2_angle — 腿2的膝关节角度（弧度）
- obs[12]: knee2_speed — 腿2的膝关节角速度（弧度/秒）
- obs[13]: leg2_contact — 腿2与地面的接触标志（1.0 接触，0.0 未接触）
- obs[14]~obs[23]: lidar_0~lidar_9 — 10个激光雷达测距仪沿地形前方的距离测量值（米或其他线性单位）

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- 每个维度连续，值域 [-1.0, 1.0]
- action[0]: hip_torque_leg1 — 施加在腿1髋关节上的力矩
- action[1]: knee_torque_leg1 — 施加在腿1膝关节上的力矩
- action[2]: hip_torque_leg2 — 施加在腿2髋关节上的力矩
- action[3]: knee_torque_leg2 — 施加在腿2膝关节上的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `reached_end_of_terrain` — 智能体成功走完整个地形
- failure-like termination: `body_fallen_over` — 身体摔倒（可能是倾斜过度或接触异常）
- ambiguous termination: 无
- truncation: 无（`step` 返回的第二个终止标志恒为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （info 为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（因为 info 为空）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前观测数组（可整体使用，或按上述索引提取）
- action：当前执行的动作数组
- next_obs：下一步观测数组
- info：仅当环境明确提供且经分析允许的字段，这里为空字典，无法提供额外信号
- training_progress：禁止使用（prompt 未明确要求使用）

禁止使用：
- original_reward（官方奖励已被屏蔽）
- official_reward
- 任何未在 observation_space 字段中明确声明的 obs 切片
- 任何 info 字段（info 为空且无保证）

## 7. 可用于奖励函数的信号
- position: 没有直接的世界坐标，但可通过速度积分推断前进位移（如累计水平速度）
- velocity: horizontal_velocity（前向速度），鼓励快速前进；vertical_velocity（垂直速度，过大的垂直振荡可能代表不稳定）
- orientation: hull_angle（身体倾角，接近0表示稳定），hull_angular_velocity（角速度，用于惩罚剧烈摇晃）
- contact: leg1_contact, leg2_contact（脚掌着地信息，可用于奖励合理的步态转换或避免长时间双脚离地/单脚着地）
- action/engine: 动作幅度（如 sum(|action|) 或 sum(action^2)）可用于惩罚过大力矩以减少能量消耗
- lidar: lidar_0..9（前方地形距离，作为潜能信号鼓励前进；但需小心使用，避免只依赖前方空旷而不鼓励实际前进）

## 8. 不确定或不可用的信号
- 世界坐标/绝对位置：未提供，无法直接获取“已行走距离”。
- 成功标志：info 为空，无显式 success/failure 标志，只能通过终止条件推断（但终止条件本身不直接出现在 reward 中）。
- 地形尽头到达信号：仅在终止时触发，不能在 reward 中预测使用，但可用于事后分析。
- 能耗的精确物理量：没有直接的能耗观测，只能通过动作幅度近似。
- 地形难度/坡度：未显式提供，只能通过 lidar 间接感知。



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