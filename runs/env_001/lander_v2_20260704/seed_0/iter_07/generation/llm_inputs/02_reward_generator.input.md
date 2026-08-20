# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
控制一个类似飞行器的刚体，从视口顶部中央附近以随机初始力开始运动，尽快到达并稳定降落在场地中央的目标着陆垫上，同时尽可能减少引擎推力消耗。智能体需要学会平滑接近目标、减小速度、保持姿态稳定，并实现两条着陆腿与垫面的安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（根据一般标准推断，源未显式声明）
- obs[0]: x_position — 相对于目标垫平面的水平坐标
- obs[1]: y_position — 相对于目标垫高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 竖直线速度
- obs[4]: body_angle — 刚体朝向角度
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑腿接触标志（0.0 或 1.0）
- obs[7]: right_support_contact — 右支撑腿接触标志（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: 无引擎推力（do nothing）
- action 1: 点燃左姿态调整引擎（产生一个方向的力矩）
- action 2: 点燃主引擎（产生推力）
- action 3: 点燃右姿态调整引擎（产生相反方向的力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  身体稳定并静止（body_not_awake_or_settled），两条支撑腿均接触着陆垫（left_support_contact == 1.0 且 right_support_contact == 1.0），且水平与垂直位置接近目标。
- failure-like termination:
  1) 身体与地面发生不当碰撞（crash_or_body_contact），例如船体非支撑部分触地；
  2) 水平位置超出视口范围（horizontal_position_outside_viewport）；
  3) 身体在未成功着陆的情况下失去活动或稳定（body_not_awake_or_settled，但没有双支撑接触或位置远离目标）。
- ambiguous termination:
  身体进入稳定静止但只有一个支撑腿接触或完全无接触，位置接近目标——可能属于不完美着陆或“搁浅”，需根据实际任务要求判断成功与否。
- truncation:
  从提供的源码看，没有环境层级的截断机制，episode 会一直运行直到触发以上终止条件。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 字典为空，无 success 标志）
- explicit_failure_flag_available: false （无 failure 标志）
- allowed_info_fields: 无（info 固定为 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

说明：判断成功或失败只能根据 next_obs 中的位置、速度、支撑接触标志以及 terminated 信号间接推断，不能依赖任何 info 字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（8 维数组，含义见观察空间定义）
- action（执行的离散动作，0~3）
- next_obs（8 维数组，transition 后的状态）
- info 中明确允许的字段（无）

禁止使用：
- original_reward （即被屏蔽的官方奖励）
- official_reward 或任何预置奖励
- 未声明的 info 字段
- 未声明的 obs / next_obs 切片含义（但 obs/next_obs 全部 8 维均已知）
- training_progress（本环境未明确允许使用，故不得使用）

## 7. 可用于奖励函数的信号
- position: next_obs[0]（水平相对位置）, next_obs[1]（垂直相对位置）
- velocity: next_obs[2]（水平速度）, next_obs[3]（垂直速度）
- orientation: next_obs[4]（角度）, next_obs[5]（角速度）
- contact: next_obs[6]（左支撑腿接触）, next_obs[7]（右支撑腿接触）
- action/engine: 动作选择（action，0~3），可用来惩罚引擎使用

## 8. 不确定或不可用的信号
- 官方奖励值（original_reward）——已屏蔽，不可用且不应重建
- 任何 info 字段（如 success, failure, termination_reason 等）——不存在
- 训练的 progress 参数——未授权使用
- 环境内部物理信息（如引擎推力大小、风力等）——未暴露在观察中，不可用



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
- best_score_so_far: -17.840

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| distance_reward + landing_quality + stability_penalty | 4 | -17.840 | -73.820 | unsolved |
| distance_reward + soft_landing_proxy + stability_penalty | 2 | -85.750 | -85.750 | unsolved |

## Previous interventions

- iter 2 (score=-85.750, structure=distance_reward + soft_landing_proxy + stability_penalty): selected_level**：Level 2 — distance_reward的数学形态（-dist_to_target恒负）对进展引导职责不合理，属于state_to_improvement变换，非单纯尺度问题。 | selected_intervention**：将distance_reward从状态值`-dist_to_target`变换为改善量`(dist_before - dist_after) * 5.0`。接近目标时获得正奖励，远离时受罚。系数5.0补偿单步距离增量量级较小的问题，使其与stability_penalty可比。
- iter 3 (score=-17.840, structure=distance_reward + landing_quality + stability_penalty): 4. selected_level: Level 2** — the `soft_landing_proxy` active_rate is near zero and its hard-binary mathematical form collapses feedback, directly matching the `sparse_to_dense` trigger: "completion/landing bonus 的 acti | 5. selected_intervention
- iter 4 (score=-110.720, structure=distance_reward + landing_quality + stability_penalty): selected_level**：Level 2 — `state_to_improvement`。证据模式直接命中"占据好状态即可持续获奖"（532的landing_quality vs 0次着陆），单靠调系数（Level 1）无法消除悬停激励的结构性漏洞。上一轮（iter 3）已通过Level 2将sparse soft_landing_proxy改为连续landing_quality并成功消除坠毁模式，但引入了新的proxy fa | selected_intervention**：将landing_quality从**持续状态值**改为**状态改善量**（potential-based shaping）：`reward = scale * (quality_potential(next_obs) - quality_potential(obs))`，使用有符号差分以避免max(0,⋅)带来的振荡套利风险，scale设为10.0使单回合总贡献上限约20（quality
- iter 5 (score=-23.320, structure=distance_reward + landing_quality + stability_penalty): `selected_level`：Level 2 — 触发条件为proxy_to_completion_alignment：状态型proxy使agent可绕过任务完成（双腿着陆）而持续获得高奖励，外部score为负而proxy奖励高；上一轮state_to_improvement失败证明不能用纯势能差替代，需要对状态型proxy做结构变换使其更贴近任务完成条件。 | `selected_intervention`：仅修改landing_quality组件，从加权和形态变为contact作为软乘数的乘法结构。将contact_score从加性因子改为软地板乘法因子`contact_factor = 0.3 + 0.7 * contact_raw`，pose部分仅保留speed/angle/angvel三因子的均值，landing_quality = 5.0 * proximity_gate * con
- iter 6 (score=-73.820, structure=distance_reward + landing_quality + stability_penalty): `selected_level`：Level 2，触发条件为 `proxy_to_completion_alignment`（proxy 提高但外部任务不升）叠加 `state_to_improvement`（占据好状态即可持续获奖）。上轮 Level 1 尺度调整（加 contact 乘子）已被 evidence 证伪，不可重复调系数。 | `selected_intervention`：将 landing_quality 从状态值奖励转换为势能改善奖励（state_to_improvement）——`landing_quality = scale * (potential(next_obs) - potential(obs))`，其中 potential 使用 proximity、soft-contact、pose 三因子的连续乘积；系数保持 5.0 以匹配新值域。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
