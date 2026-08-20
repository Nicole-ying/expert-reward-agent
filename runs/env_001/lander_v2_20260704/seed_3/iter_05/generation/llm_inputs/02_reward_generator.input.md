# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
控制一个二维飞行器，从初始位置出发，尽快到达并稳定降落在视口中央的目标平台上，同时尽量减少引擎推力消耗。飞行器初始带有随机微小作用力，需要学会接近目标、减速、保持稳定姿态、并安全着陆（实现两侧支撑脚同时接触，且速度接近于零）。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（根据典型环境设定，第6、7维为0/1二值，但整体浮点）
- obs[0]: x_position — 机体相对于目标平台中心的水平坐标（m）
- obs[1]: y_position — 机体相对于目标平台高度的垂直坐标（m）
- obs[2]: x_velocity — 机体水平线速度（m/s）
- obs[3]: y_velocity — 机体垂直线速度（m/s）
- obs[4]: body_angle — 机体倾斜角度（rad）
- obs[5]: angular_velocity — 机体角速度（rad/s）
- obs[6]: left_support_contact — 左侧支撑脚接触标志（1.0 接触, 0.0 未接触）
- obs[7]: right_support_contact — 右侧支撑脚接触标志（1.0 接触, 0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine — 不点火，仅靠惯性飞行
- action 1: left_orientation_engine — 点燃左侧姿态调整引擎（产生使机体逆时针旋转的力矩）
- action 2: main_engine — 点燃主引擎（产生向上的推力）
- action 3: right_orientation_engine — 点燃右侧姿态调整引擎（产生使机体顺时针旋转的力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled — 如果同时满足位置接近目标、速度极小、姿态平稳且两侧支撑脚接触，则很可能代表成功着陆
- failure-like termination: crash_or_body_contact — 机体与地面或其他非目标部分发生碰撞；horizontal_position_outside_viewport — 水平位置超出视野边界
- ambiguous termination: body_not_awake_or_settled — 也可能只是漂移到不可控状态进入沉睡，未必成功，需结合 position/contact/velocity 判断
- truncation: 信息中未提供 episode 长度限制，但一般环境会有默认 step 上限

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 该环境 step 返回 info 为空字典 {}，因此只能使用 obs、next_obs、action，不得使用 info
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；original_reward 不可用；任何未在 obs 或 next_obs 中明确定义的信号均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs (numpy array, shape (8,))
- action (int, 取值 0/1/2/3)
- next_obs (numpy array, shape (8,))
- info 中的字段：无（info为空字典）
- training_progress：该 prompt 未明确允许，不应使用

禁止使用：
- original_reward（官方奖励被 mask）
- 任何 info 字段（不存在）
- 任何未在上述 obs 中声明的信号（如 true_success 等隐藏变量）

## 7. 可用于奖励函数的信号
- position: next_obs[0]（x 误差）、next_obs[1]（y 误差），可构造距离奖励/惩罚
- velocity: next_obs[2], next_obs[3] 可惩罚着陆时速度过快
- orientation: next_obs[4] 可惩罚非水平姿态
- angular_velocity: next_obs[5] 可惩罚旋转
- contact: next_obs[6], next_obs[7] 可鼓励双足同时接触
- action/engine: action 值可用于惩罚引擎使用（尤其 1,2,3 相对于 0 有额外消耗）
- 复合信号：可组合位置、速度、姿态、接触构建成功着陆判断（如 |x|<ε_x, y≈0, |v|<ε_v, |angle|<ε_angle, left_contact & right_contact 均为 1）

## 8. 不确定或不可用的信号
- 显式 success / failure 标记（info 中未提供）
- 目标绝对坐标（仅提供相对坐标，但已足够）
- 风力数值、引擎推力大小（被遮蔽）
- 任何由官方奖励衍生出的中间信号（不可用）



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架是任务相关的设计原语、风险提示和参考起点，不构成封闭候选集合。可直接采用、组合、变形，或基于环境事实提出新的数学结构。

## 1. 任务路由摘要
- navigation_goal_reaching：任务目标是接近/到达指定位置。重点观察 goal_near_oscillation / high_reward_without_success / fast_crash_near_goal。

## 2. 相关奖励骨架摘要（按任务路由检索）

以下骨架由任务路由检索推荐。是否使用某个骨架取决于：
1. 该骨架所需信号是否在环境接口中实际可用；
2. 是否与该任务阶段匹配（v1 优先设计核心学习信号，效率/安全类后续迭代加入）。

### progress_delta_reward
- 角色: 密集学习引导
- 数学形态: d(obs, goal) - d(next_obs, goal)
- 需要信号: obs[0], obs[1], next_obs[0], next_obs[1]
- 使用说明: 奖励每一步更接近目标。适合目标位置已知的导航/到达任务。
- 风险: 目标附近震荡。
- 后续迭代: 可 clip；后续配合成功、时间或稳定信号。

### distance_reward
- 角色: 密集过程引导
- 数学形态: -d(obs, goal)
- 需要信号: obs[0], obs[1]
- 使用说明: 连续负距离信号，引导 agent 靠近目标。与 progress_delta_reward 同时大权重使用会产生重复信号。
- 风险: 接近目标但不完成；不关心速度和姿态。
- 后续迭代: 训练后检查 high_reward_without_success。

### potential_based_shaping
- 角色: 势能塑形
- 数学形态: gamma*Phi(next_obs)-Phi(obs)
- 需要信号: 可定义 Phi
- 使用说明: 基于势能差分的塑形信号。比 progress_delta 更抽象，当任务有明确的势能定义时可使用。
- 风险: Phi 错误会误导学习。
- 后续迭代: 如果需要更标准的 shaping，再替换或补充。

### stability_penalty
- 角色: 轻量稳定约束
- 数学形态: -lambda_v*|velocity| - lambda_a*|angle| - lambda_w*|angular_velocity|
- 需要信号: next_obs[2], next_obs[3], next_obs[4], next_obs[5]
- 使用说明: 抑制高速、大角度或高角速度。适合需要稳定运动或姿态控制的任务。
- 风险: 过强会保守或不敢动。
- 后续迭代: 若高速失稳，增大权重。

### soft_landing_proxy
- 角色: 任务完成近似信号
- 数学形态: small_bonus if near_target and low_speed and stable_angle and both_contact else 0
- 需要信号: position, velocity, angle, contact flags
- 使用说明: 多条件组合的完成近似。不能直接把 contact 当 success。
- 风险: 如果条件太宽，会变成 contact reward hacking。
- 后续迭代: 如果 high_reward_without_success，收紧条件或移除。

### terminal_success_reward
- 角色: 任务目标奖励
- 数学形态: R_success * I[success]
- 需要信号: 显式 success flag
- 使用说明: 当环境提供显式 success flag 时可用。若 explicit_success_flag_available=false，不可使用。
- 风险: 会诱导 LLM 发明 info['success']。
- 后续迭代: 当 wrapper 明确暴露 success 后再加。

### terminal_failure_penalty
- 角色: 失败惩罚
- 数学形态: -R_failure * I[failure]
- 需要信号: 显式 failure flag 或 termination_reason
- 使用说明: 当环境提供显式 failure flag 时可用。若 explicit_failure_flag_available=false，不可使用。
- 风险: 误判终止原因。
- 后续迭代: 当能区分失败终止后再加。

### time_penalty
- 角色: 效率约束
- 数学形态: -lambda_time
- 需要信号: 每步调用
- 使用说明: 惩罚每步耗时。先完成任务再加入，不建议 v1 使用。
- 风险: 可能导致冒险或快速失败。
- 后续迭代: 若能接近但拖太久，再小权重加入。

### energy_penalty
- 角色: 动作/能耗约束
- 数学形态: -lambda_action * engine_use(action)
- 需要信号: action
- 使用说明: 惩罚动作幅度/能耗。先完成任务再加入，v1 太早加入可能不敢动。
- 风险: agent_afraid_to_move。
- 后续迭代: 能完成任务并稳定后再优化能耗。

### gated_reward
- 角色: 安全门控
- 数学形态: if unsafe then penalty else task_reward
- 需要信号: 明确 unsafe 条件
- 使用说明: 按条件切换奖励模式。v1 不建议使用复杂门控。
- 风险: 门控过严导致学不到。
- 后续迭代: 若安全被进度奖励抵消，再加入。

## 3. reward_v1 生成要求
- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget：每个组件必须有明确角色，不能为了显得完整而堆叠。
- 将上述骨架作为思考面而非允许列表；最终设计由任务目标、可用信号和预期策略行为决定，不要求组件对应已有 skeleton_id。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。



# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: 149.440

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| approach_reward + soft_landing_proxy + stability_penalty | 4 | 149.440 | 136.730 | unsolved |

## Previous interventions

- iter 2 (score=149.440, structure=approach_reward + soft_landing_proxy + stability_penalty): `selected_level`：Level 1——stability_penalty的数学形态（加权绝对值和）本身合理，证据主要指向其系数过强（|penalty/progress| ≈ 1.09 > 0.5），先做尺度修复。 | `selected_intervention`：仅将stability_penalty三个系数全部降低10倍——w_speed: 0.01→0.001, w_angle: 0.1→0.01, w_angvel: 0.05→0.005，使|penalty/progress|目标值≈0.11，接近轻约束起点0.1。
- iter 3 (score=-10.510, structure=approach_reward + soft_landing_proxy + stability_penalty): `selected_level`：Level 1。approach_reward 的职责、符号和数学形态均合理，证据仅指向其相对 soft_landing_proxy 过弱（累计比 ≈1:67），不需要结构变换。 | `selected_intervention`：将 approach_reward 乘以系数 25.0，其他组件不变。预期总 approach_reward 从 ~1.07 升至 ~26.75，约为 soft_landing_proxy 的 37%，使进度信号在早期学习中具备实际引导力。
- iter 4 (score=136.730, structure=approach_reward + soft_landing_proxy + stability_penalty): `selected_level`：Level 1。soft_landing_proxy 在 best 轮（iter 2）中曾正常激活（sum=1.728），说明其数学形态在合理 approach 尺度下可达；当前 active_rate=0 是 25x approach 的后果，不是 proxy 结构的固有缺陷。 | `selected_intervention`：仅修改 approach_reward 系数，从 25.0 降至 3.0。best 的 1.0 已验证可行，25.0 已验证失败，3.0 是介于两者之间的中等强度，保留"尽可能快到达目标"的任务激励同时避免压倒着陆信号。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
