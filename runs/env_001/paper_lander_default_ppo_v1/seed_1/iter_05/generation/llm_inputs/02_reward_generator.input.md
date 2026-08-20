# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器，从初始随机位置/速度出发，尽快到达并稳定地降落在画面中央的目标垫上，同时尽量减少引擎推力的使用。在接近目标时需要减速、保持稳定的姿态并安全地触垫。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测，包含浮点数与 0.0/1.0 标志）
- obs[0]: horizontal position relative to target pad (x_position_relative_to_target)
- obs[1]: vertical position relative to pad height (y_position_relative_to_pad_height)
- obs[2]: horizontal linear velocity (x_velocity)
- obs[3]: vertical linear velocity (y_velocity)
- obs[4]: body orientation angle (body_angle)
- obs[5]: angular velocity (angular_velocity)
- obs[6]: left support contact flag (1.0 = contact, 0.0 = no contact)
- obs[7]: right support contact flag (1.0 = contact, 0.0 = no contact)

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no engine – 不启动任何引擎，滑行/惯性飞行
- action 1: left orientation engine – 启动左侧姿态调整引擎（产生转向力矩）
- action 2: main engine – 启动主引擎（产生与飞行器朝向相关的推力）
- action 3: right orientation engine – 启动右侧姿态调整引擎（产生相反方向的转向力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`（机体静止/稳定）可能是成功降落在目标垫上的信号，但前提是位置已接近目标，否则也可能是其他原因导致的静止（如飘走后静止在远处）。该条件本身不直接等于成功。
- failure-like termination: 
  - `crash_or_body_contact`（坠毁或危险的身体接触）明显是失败状态。
  - `horizontal_position_outside_viewport`（水平位置超出画面边界）属于失败/失控。
- ambiguous termination: 
  - `body_not_awake_or_settled` 在没有位置信息辅助时无法明确区分成功或失败。
- truncation: 暂无明确的最大步数限制，`step` 源码中仅返回 `terminated`，没有 `truncated` 返回（默认 False）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典）
- explicit_failure_flag_available: false （info 为空字典）
- allowed_info_fields: 无（info = {}，不得使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 执行动作前的观察
- action: 执行的动作
- next_obs: 执行动作后的观察

禁止使用：
- original_reward（已被掩码，不得使用）
- info 中的任何字段（info 为空）
- training_progress（除非任务描述显式要求，此处不要求）

## 7. 可用于奖励函数的信号
- position: next_obs[0]（距目标垫的水平距离）、next_obs[1]（距目标垫的垂直高度）
- velocity: next_obs[2]（水平速度）、next_obs[3]（垂直速度）
- orientation: next_obs[4]（机体角度）、next_obs[5]（角速度）
- contact: next_obs[6]（左支撑触垫）、next_obs[7]（右支撑触垫）
- action/engine: action 的选择（0~3），可用来惩罚引擎使用

## 8. 不确定或不可用的信号
- 终止时的成功标志：不存在（info 未提供 success / failure 信号）
- 原始奖励：已被掩码，不可使用
- 终止原因字符串：不可用
- 任何环境内部的时间步数、燃料消耗等未暴露的信息：不可用



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
- best_score_so_far: -92.000

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| progress_reward + soft_landing_bonus + stability_penalty | 4 | -92.000 | -92.000 | unsolved |

## Previous interventions

- iter 2 (score=-106.270, structure=progress_reward + soft_landing_bonus + stability_penalty): 4. selected_level: Level 1**：三个组件的数学形态与职责符号均合理，问题明确是stability_penalty系数过强导致尺度失衡（|penalty/progress|=4.15，远超0.5的经验触发线）。 | 5. selected_intervention
- iter 3 (score=-107.990, structure=progress_reward + soft_landing_bonus + stability_penalty): `selected_level`: Level 2 — `sparse_to_dense` transformation of soft_landing_bonus, triggered by active_rate < 1% with all episodes crashing before landing. | `selected_intervention`: replace the hard 6-condition binary `soft_landing_bonus` with a continuous product of bounded proximity, speed, and angle scores. Each factor uses `max(0, 1 - value/threshold)` so partial improve
- iter 4 (score=-92.000, structure=progress_reward + soft_landing_bonus + stability_penalty): `selected_level`：Level 2 — 触发条件为`product_to_noncollapsing_joint`：三个[0,1]因子乘积导致奖励几乎恒塌缩至零，单因子改善无法产生有意义反馈。 | `selected_intervention`：将soft_landing_bonus从乘积 `proximity_score * speed_score * angle_score * 2.0` 改为几何平均 `(proximity_score * speed_score * angle_score) ** (1.0/3.0) * 2.0`。保持三个bounded因子和系数不变。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
