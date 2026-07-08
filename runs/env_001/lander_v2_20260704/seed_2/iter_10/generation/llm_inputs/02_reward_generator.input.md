# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器着陆任务。飞行器从靠近视口顶部中央的位置出发，受到初始随机力。目标是最快速度到达中央目标着陆区并平稳停稳，同时尽可能少使用引擎推力。智能体需要学会接近目标、减速、保持稳定姿态并安全着陆（接触着陆区）。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64（按一般 Box 推测）
- obs[0]: x_position —— 飞行器相对于目标着陆区中心的水平坐标
- obs[1]: y_position —— 飞行器相对于着陆区高度的垂直坐标（越接近 0 表示降落到着陆面高度）
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 飞行器朝向角（可能相对于竖直方向，0 为竖直向上）
- obs[5]: angular_velocity —— 角速度
- obs[6]: left_support_contact —— 左侧支撑接触标志（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact —— 右侧支撑接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine —— 不点火，无推力
- action 1: left_orientation_engine —— 点燃一个方向引擎（产生侧向/旋转推力，可能用于调整姿态）
- action 2: main_engine —— 点燃主引擎（产生主要推力，通常用于减速或悬停）
- action 3: right_orientation_engine —— 点燃另一个方向引擎（与 action 1 相对）

## 5. step 与终止条件分析
### 5.1 终止模式
terminated 在以下任一条件为真时成立：
1. **crash_or_body_contact**：可能代表机体某部位与地面/障碍物发生危险碰撞（非着陆区接触），通常为失败。
2. **horizontal_position_outside_viewport**：水平位置超出视口边界，通常为失败。
3. **body_not_awake_or_settled**：机体处于非唤醒状态（可能已稳定停止）。这一条件可能包含两种情况：① 成功着陆并稳定；② 机体掉出视口后进入睡眠。因此该终止状态本身**不直接指示成功或失败**。

- success-like termination: 当 `body_not_awake_or_settled` 为真，且同时满足：位置接近目标 (x ≈ 0, y ≈ 0)，速度接近 0，至少一侧接触标志为 1.0 时，很可能为成功着陆。但无法仅从该 flag 直接判定。
- failure-like termination: `crash_or_body_contact`、`horizontal_position_outside_viewport`。
- ambiguous termination: `body_not_awake_or_settled`（需要结合观测才能推断是否为成功）。
- truncation: 代码中未见时间截断，终止全部由 terminated 驱动。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 没有可用的 info 字段（step 返回的 info 为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为 info 为空

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测）
- action（动作）
- next_obs（下一观测）
- info 中没有可用字段，所以实质上不可使用 info

禁止使用：
- original_reward（原始奖励被屏蔽，严禁引入）
- official_reward（任何形式的官方奖励）
- training_progress（prompt 未明确允许）
- info 中未声明的字段（全部禁止）
- 任何未在观测空间中声明的 obs 切片（全部 8 个索引均已声明，可使用）

## 7. 可用于奖励函数的信号
- position: obs[0] (x), obs[1] (y) —— 相对目标的水平/垂直位置，可用于接近奖励
- velocity: obs[2] (vx), obs[3] (vy) —— 可用于速度惩罚，鼓励软着陆
- orientation: obs[4] (angle), obs[5] (angular vel) —— 可用于姿态惩罚，鼓励稳定竖直
- contact: obs[6], obs[7] —— 着陆接触标志，可用于判断是否成功着地
- action/engine: action ∈ {0,1,2,3} —— 可用于燃料消耗惩罚（非零动作施加负奖励）
- 下一观测 next_obs 中的对应信号也可使用（如位置变化、速度变化等）

## 8. 不确定或不可用的信号
- 显式的成功/失败标志（info 中无此字段）
- crash 具体类型（无法区分是碰撞还是单纯躯体接触）
- 原始的 official_reward（被屏蔽，不可用）
- 任何未在观测空间中定义的变量（如环境内部物理量）



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
- best_score_so_far: -33.720

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| distance_reward + landing_quality + stability_penalty | 7 | -33.720 | -111.980 | unsolved |
| progress_delta + weighted_stability_penalty | 1 | -109.220 | -109.220 | unsolved |
| distance_reward + soft_landing_bonus + stability_penalty | 1 | -110.970 | -110.970 | unsolved |

## Previous interventions

- iter 2 (score=-112.960, structure=distance_reward + landing_quality + stability_penalty): 4. selected_level: Level 2** — the 0.5% active_rate binary bonus matches the `sparse_to_dense` evidence pattern exactly; a coefficient tweak cannot fix structural sparsity. | 5. selected_intervention
- iter 4 (score=-109.790, structure=distance_reward + landing_quality + stability_penalty): selected_level**：Level 1 —— 稳定性惩罚的职责和数学形态合理，但尺度过强阻碍早期探索和必要机动。上一轮修改了landing_quality形态（已生效），本轮不应重复修改同一组件；stability_penalty系数尚未被调整过，且ratio虽低于0.5但处于crash场景下的"过度约束"状态。 | selected_intervention**：仅调整stability_penalty的四个权重系数，将其整体幅度降至约40%（w_vx: 0.15→0.06, w_vy: 0.05→0.02, w_angle: 0.2→0.08, w_angvel: 0.2→0.08）。distance_reward和landing_quality保持不变。
- iter 6 (score=-33.720, structure=distance_reward + landing_quality + stability_penalty): `selected_level`：Level 2——着陆信号缺失属于必要职责不完备，且Level 1单纯降稳定性系数无法弥补缺失的终端引导。 | `selected_intervention`：以best代码为基础，将landing_quality从乘积几何平均(product^0.2)改为和基联合满足(sum_of_factors)，保留distance_reward和light stability_penalty不变。
- iter 7 (score=-108.690, structure=distance_reward + landing_quality + stability_penalty): selected_level**：Level 1。landing_quality 与 distance_reward 量级抵消，且 landing_quality 过大是形成舒适区的直接原因，应通过系数调整重建相对梯度，不改变数学形态。 | selected_intervention**：仅将 landing_quality 系数从 0.2 降至 0.1，其他组件不变。
- iter 8 (score=-34.070, structure=distance_reward + landing_quality + stability_penalty): 4. selected_level: Level 2：同一骨架家族已在 iter 3/4/6/7 迭代 4 轮，3/4 次得分为 -108 左右，仅 iter 6 偶然成功，说明当前数学结构存在系统性训练不稳定。需要对 distance_reward 做 `unbounded_to_bounded` 结构变换，将线性无界惩罚改为有界饱和形式，从结构上消除"冲撞是局部最优"的问题。 | 5. selected_intervention
- iter 9 (score=-111.980, structure=distance_reward + landing_quality + stability_penalty): 4. `selected_level`：Level 2——`state_to_improvement`变换。证据直接表明持久状态奖励被利用（长episode、高landing_quality累积、低外部进展），且上轮指数距离修改未消除悬停行为，需要结构变换而非尺度调整。 | 5. `selected_intervention`：将landing_quality从状态值`0.2 * sum_of_factors(next_obs)`变换为势能差`5.0 * (sum_of_factors(next_obs) - sum_of_factors(obs))`，系数5.0匹配改善形式的值域（单episode最大改善约5.0→总贡献约25，与distance/stability可比较）。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
