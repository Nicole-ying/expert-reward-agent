# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
这是一个 2D 着陆器任务。一个类似飞行器的主体从视口顶部中央出发，受到一个随机初始力。
核心目标：尽可能**快速**且**稳定**地降落在中央目标平台上，同时**尽量减少发动机推力**（燃料消耗）。  
智能体需要学会接近目标、减速、保持姿态平衡，并安全接触平台。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (默认，接触标志以 1.0/0.0 表示)
- obs[0]: x_position — 飞行器相对目标平台的水平坐标（中心为 0）
- obs[1]: y_position — 飞行器相对平台高度的垂直坐标（平台高度处为 0）
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体倾斜角度
- obs[5]: angular_velocity — 机体角速度
- obs[6]: left_support_contact — 左支撑腿接触标志（接触平台为 1.0，否则 0.0）
- obs[7]: right_support_contact — 右支撑腿接触标志（接触平台为 1.0，否则 0.0）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，无推力
- action 1: left_orientation_engine — 点燃左侧姿态发动机（产生使机体向左旋转的力矩）
- action 2: main_engine — 点燃主发动机（产生向上的推力）
- action 3: right_orientation_engine — 点燃右侧姿态发动机（产生使机体向右旋转的力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled — 当机体不再活跃或达到稳定状态时终止。如果此时机体已位于目标平台上、姿态平稳、双腿接触，可视为成功着陆。
- failure-like termination: crash_or_body_contact — 机体主体碰撞地面或接触平台之外的表面（非支撑腿的正常接触）。通常意味着坠毁或严重撞击。
- failure-like termination: horizontal_position_outside_viewport — 机体水平飞出视野，无法返回。
- ambiguous termination: 无明确标记的中间状态。终止时只有观测数据，无额外成功/失败标签。
- truncation: 本环境没有明确的时间截断（step 返回的第四个值是 False，即 truncated 恒为 False），但多数实际包装中会附加时间上限。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （无，info 为空字典，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；不能假设 info["success"] 或 info["failure"] 存在。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs – 当前观测（8维数组）
- action – 当前动作（0~3 整数）
- next_obs – 下一时刻观测（8维数组）
- info – 空字典，不允许从中读取任何字段
- training_progress – 训练进度（0.0~1.0），仅在 prompt 明确允许时使用

禁止使用：
- original_reward – 原始奖励被屏蔽，严禁直接使用
- official_reward – 同 original_reward
- info 中的任何未声明字段（info 目前为空）
- obs 中未在“可用于奖励函数的信号”里声明的切片

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1]) — 用于计算到目标平台的距离、靠近程度
- velocity: x_velocity (obs[2]), y_velocity (obs[3]) — 用于衡量运动速度、鼓励减速
- orientation: body_angle (obs[4]), angular_velocity (obs[5]) — 用于评估姿态稳定性和旋转惩罚
- contact: left_support_contact (obs[6]), right_support_contact (obs[7]) — 双腿是否接触平台，用于判断着陆状态
- action/engine: action 本身 — 可基于用了哪些引擎（特别是主发动机）给予燃料消耗惩罚

## 8. 不确定或不可用的信号
- 无显式的成功/失败标志（info 为空，无法直接获取 success、failure、termination_reason）
- 目标的绝对距离需要从 x_position、y_position 自行计算（无直接距离传感器）
- 燃料剩余、时间步、风力等环境内部状态不可见
- 原始奖励 original_reward 被屏蔽，不可用



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
- best_score_so_far: 108.450

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| approach_landing_reward + distance_cost + stability_cost | 4 | 108.450 | 108.450 | unsolved |
| distance_cost + soft_landing_bonus + stability_cost | 1 | -119.560 | -119.560 | unsolved |
| approach_landing_reward + distance_cost + engine_penalty + stability_cost | 1 | -125.740 | -125.740 | unsolved |

## Previous interventions

- iter 2 (score=-98.890, structure=approach_landing_reward + distance_cost + stability_cost): 4. `selected_level`：Level 2 — `sparse_to_dense`：完成 bonus 的结构性稀疏（0.3% << 1%）明确要求将硬性二值事件替换为连续过程证据，而非仅调系数。 | 5. `selected_intervention`：将 `soft_landing_bonus` 从硬性六条件联合判定，转换为由接近度、速度、姿态角和支撑腿接触的连续有界指数得分之和构成的 `approach_landing_reward`。
- iter 3 (score=-87.910, structure=approach_landing_reward + distance_cost + stability_cost): `selected_level`：Level 2 — `dense_to_task_event`。稠密代理已教会agent到达着陆垫（ep长度从68→752，不再crash），但现在主导奖励并阻止高效完成。证据模式完全匹配："dense proxy forming medium score plateau"和"proxy提高但外部任务不升"。根据知识库指引：应reduce/difference/localize该dense proxy。 | `selected_intervention`：将approach_landing_reward从**持续性状态值**（state-value：`e^(-dist²)`、`e^(-vel²)`、contact连续值）重构为**改善量**（improvement-based：`max(0, prev - curr)`形式），并将velocity improvement以proximity gate约束（仅近端减速获奖励）。四个子项统一改为
- iter 5 (score=108.450, structure=approach_landing_reward + distance_cost + stability_cost): selected_level**: Level 1 — distance_cost role and monotonic math form are reasonable, but its coefficient makes penalty dominate progress at 1.94≫0.5, a clear scale problem per penalty_magnitude_control evidence. | selected_intervention**: Reduce w_dist from 1.0 to 0.08 (12.5× reduction, exceeding the 10× minimum for penalty dominance), applied to best code (iter 3 base with proximity_improvement=5.0), no other component changed.
- iter 6 (score=-125.740, structure=approach_landing_reward + distance_cost + engine_penalty + stability_cost): `selected_level`：Level 2。触发条件：improvement型proxy（proxy_to_completion_alignment模式）被策略系统性套利——外部任务完成（成功着陆）但效率极差（episode过长），且缺失引擎使用约束这一任务明确要求。 | `selected_intervention`：新增`engine_penalty`组件——对任何非零动作（action 1/2/3使用引擎）施加固定惩罚，惩罚"使用引擎推力"这一行为本身。这是当前奖励完全缺失的任务维度。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
