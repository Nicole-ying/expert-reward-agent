# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
这是一个二维飞行器轨迹优化任务。飞行器从视口顶部附近开始，受到随机初始力。它必须尽快到达并稳定地降落在中央目标平台上，同时尽可能少地使用发动机推力。智能体需要学会靠近目标、减速、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（假设，环境未指定但通常如此）
- obs[0]: x_position，飞行器相对于目标垫的水平坐标
- obs[1]: y_position，飞行器相对于平台高度的垂直坐标
- obs[2]: x_velocity，水平线速度
- obs[3]: y_velocity，垂直线速度
- obs[4]: body_angle，机体方向角
- obs[5]: angular_velocity，角速度
- obs[6]: left_support_contact，左侧支撑腿接触标志（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact，右侧支撑腿接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，不启动任何引擎（滑行）
- action 1: left_orientation_engine，启动左侧姿态引擎
- action 2: main_engine，启动主引擎（向下喷气产生向上推力）
- action 3: right_orientation_engine，启动右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 且飞行器位置在目标平台上方，左右支撑腿均接触，速度接近零。这表示安全着陆并稳定。
- failure-like termination: crash_or_body_contact（机体非腿部部分与地面或其他障碍物碰撞），或者 horizontal_position_outside_viewport（水平飞出视口边界）。
- ambiguous termination: body_not_awake_or_settled 也可能由坠毁后静止触发，此时需结合位置和接触腿信息判断。
- truncation: 环境未提及时间截断，但可能由封装器实现（本描述不涉及）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 字典为空）
- forbidden_or_uncertain_info_fields: 任何 info 字段（包括 potential success、failure、termination_reason 等）均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前观察（8维数组）
- action：当前执行的动作（int）
- next_obs：下一时刻观察（8维数组）
- info 中明确允许的字段：无（info 始终为空字典）
- training_progress：只有在 prompt 明确允许时才使用（此处未明确允许，避免使用）

禁止使用：
- original_reward（官方奖励被屏蔽，不得参考或使用）
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片

## 7. 可用于奖励函数的信号
- position: obs[0]（水平坐标）、obs[1]（垂直坐标）
- velocity: obs[2]（水平速度）、obs[3]（垂直速度）
- orientation: obs[4]（机体角度）、obs[5]（角速度）
- contact: obs[6]（左腿接触）、obs[7]（右腿接触）
- action/engine: action 的取值（0~3），可推断推力使用

## 8. 不确定或不可用的信号
- 明确的成功/失败标志：不存在
- 目标是否已到达：只能通过位置、速度、接触等信息间接推断
- 剩余燃料量：未提供
- 时间/步数计数：未作为观察给出
- 外部风速或随机力：不可观测



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
- best_score_so_far: -10.240

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| distance_anchor + progress_reward + soft_landing_near + stability_penalty | 1 | -10.240 | -10.240 | unsolved |
| distance_anchor + progress_reward + soft_landing_contact_gated + stability_penalty | 4 | -17.330 | -17.330 | unsolved |
| distance_anchor + progress_reward + soft_landing_bonus + stability_penalty | 1 | -116.830 | -116.830 | unsolved |
| distance_anchor + progress_reward + soft_landing_proximity + stability_penalty | 2 | -187.420 | -196.870 | unsolved |
| shaping_reward + stability_penalty | 1 | -591.150 | -591.150 | unsolved |

## Previous interventions

- iter 2 (score=-187.420, structure=distance_anchor + progress_reward + soft_landing_proximity + stability_penalty): 4. **selected_level**：Level 2 — sparse_to_dense。证据直接否定当前二元合取形态（0.4% active_rate），且上一轮首次设计尚未做 Level 2 变换。 | 5. **selected_intervention**：仅修改 soft_landing_bonus 组件，从稀疏二元合取变换为连续有界乘积代理信号 `soft_landing_proximity`，使用 `max(0, 1 - x/threshold)` 对各维度提供梯度，并保持相同权重 0.5。
- iter 3 (score=-110.470, structure=distance_anchor + progress_reward + soft_landing_contact_gated + stability_penalty): 4. **selected_level**：Level 2 — `dense_to_task_event`。证据直接否定当前无门控连续形态：连续代理在未接触时激活，得分反比稀疏二元版本差70分，需用接触门控将奖励对齐到任务完成条件。 | 5. **selected_intervention**：唯一修改soft_landing组件，改为**接触门控的连续着陆奖励**——仅当`left_contact>0.5`或`right_contact>0.5`时才激活连续梯度乘积，同时放宽阈值（dist→1.0, vel→0.5, angle→0.3, angvel→0.3）使着陆条件比Iter 1更可达，w_landing提至1.0补偿门控后的稀疏性。
- iter 4 (score=-110.830, structure=distance_anchor + progress_reward + soft_landing_contact_gated + stability_penalty): 4. selected_level: Level 2：`global_to_local_gate`。stability_penalty的数学形态对接近阶段约束不足，证据直接表明agent在目标附近高速crash。该组件职责正确（安全约束），但需要从全局均匀变为接近平台时显著增强。符合"约束在无关阶段妨碍探索"的证据模式。 | 5. selected_intervention
- iter 6 (score=-196.870, structure=distance_anchor + progress_reward + soft_landing_proximity + stability_penalty): `selected_level`：Level 2 — 触发条件为`sparse_to_dense`模式：完成信号因接触门控几乎不触发（episode_sum_mean=0.008），Agent在接触到平台前就已失败，需要将稀疏事件奖励转化为连续过程证据。 | `selected_intervention`：唯一目标组件为`soft_landing_contact_gated`，将其从接触门控（`if contact_active:`）改为距离门控（`dist_factor`自动在dist>threshold时归零），保持乘积形态不变（dist_factor × vel_factor × angle_factor × angvel_factor × contact_boost），使Agent在
- iter 7 (score=-71.150, structure=distance_anchor + progress_reward + soft_landing_contact_gated + stability_penalty): 4. `selected_level`：Level 2 — `product_to_noncollapsing_joint`。证据直接表明乘积塌缩（active_rate=0.6%），不是单纯尺度问题。 | 5. `selected_intervention`：从best代码（iter3，接触门控版）出发，将soft_landing_contact_gated内部四个因子的**乘积**改为**算术平均**，即`dist_factor * vel_factor * angle_factor * angvel_factor` → `(dist_factor + vel_factor + angle_factor + angvel_factor)
- iter 8 (score=-17.330, structure=distance_anchor + progress_reward + soft_landing_contact_gated + stability_penalty): `selected_level`：Level 1。distance_anchor的职责（持续拉力）、符号（负值惩罚距离）和数学形态（线性）均合理，证据主要表明该组件过强（|anchor|/progress ≈ 1.2 > 0.5），且本骨架家族尚未做过尺度修复。 | `selected_intervention`：将w_dist从0.1降至0.01（10倍衰减），使distance_anchor从主导惩罚退化为轻量约束。其他所有参数保持不变。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
