# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
控制一个近似车辆的刚体，从画面顶部中央附近出发，尽快到达并稳定降落在正中央的目标垫上，同时尽量少用引擎推力。  
智能体需要学会：
- 飞向目标区域；
- 降低速度、保持平稳姿态；
- 安全触垫并最终静止。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 要求移动到指定的目标垫位置并稳定停留，核心为导航型目标达成任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推测为 float32（具体由环境内部实现确定）
- obs[0]: x_position — 相对目标垫中心的水平距离
- obs[1]: y_position — 相对垫上方某参考高度的垂直距离
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体方向角
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑/接触标志（1.0 表示接触，0.0 表示未接触）
- obs[7]: right_support_contact — 右支撑/接触标志（1.0 表示接触，0.0 表示未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: 无引擎 — 不点火，仅靠惯性运动
- action 1: 左姿态引擎 — 点燃一个方向调整引擎（提供姿态修正推力）
- action 2: 主引擎 — 点燃主推进引擎（通常提供向上的推力）
- action 3: 右姿态引擎 — 点燃另一个方向调整引擎（与左姿态引擎相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 机体稳定且不再活动（`body_not_awake_or_settled` 为真），且未发生 crash 或越界。暗含左右接触点可能均已触垫，并处于静止状态。
- failure-like termination: 发生 crash 或与地面/非垫区域接触（`crash_or_body_contact`），或水平位置超出视口（`horizontal_position_outside_viewport`）。
- ambiguous termination: 机体休眠但可能并非在目标垫上（例如飘离很远后因外力停止），仅靠终止信号无法区分。
- truncation: 本环境不返回截断信号（step 返回 `False`），无基于步数的超时终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] (step 返回的 info 为空字典)
- forbidden_or_uncertain_info_fields: 任何 info 键均不允许使用；试图从 info 中读取 success/failure/termination_reason 等字段均不可靠，源码中不存在。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
参数约束：
- 允许使用：
  - `obs`：当前观察向量（8 维）
  - `action`：刚执行的动作索引（0~3）
  - `next_obs`：下一观察向量（8 维）
  - `info`：空字典，禁止依赖任何字段
  - `training_progress`：仅在 prompt 明确要求时才使用
- 禁止使用：
  - `original_reward`（即被遮盖的官方奖励值）
  - 任何未在 `obs`/`next_obs` 中定义的维度
  - info 的任意键
  - 环境内部隐藏状态或真实终止原因标志

## 7. 可用于奖励函数的信号
- 位置：x_position, y_position（可直接鼓励接近目标区域，如惩罚距离）
- 速度：x_velocity, y_velocity（可用于奖励减速、降低动能）
- 姿态：body_angle（鼓励保持竖直/水平的稳定姿态）
- 角速度：angular_velocity（抑制旋转）
- 接触：left_support_contact, right_support_contact（奖励安全触垫、惩罚单侧或过早接触）
- 动作/引擎：action 类型（可对燃料消耗进行惩罚，如对引擎点火的额外代价）
- 此外，`next_obs` 可与 `obs` 对比获得相邻状态变化。

## 8. 不确定或不可用的信号
- **明确的成功/失败标志**：info 中不存在，不可用；终止时为自主推导。
- **机体的“休眠”状态**：虽然作为终止条件之一，但无法在奖励函数中直接获取。只能通过观察向量中静止特征（速度≈0，接触标志为1）间接推测。
- **外部风的干扰**：源码中提到的 wind 已剔除，无法感知。
- **相对目标垫的精确距离**：obs 中给出的是坐标，未提供直接的 L2 距离，需自行计算。
- **游戏物理参数**：质量、推力大小、重力等均不可见。



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。

## 1. 任务路由摘要
- navigation_goal_reaching：用密集过程引导；无 success flag 时禁用终点成功核心项；重点观察 goal_near_oscillation / high_reward_without_success / fast_crash_near_goal。

## 2. 相关奖励骨架摘要
### progress_delta_reward
- 角色: 主学习引导
- 数学形态: d(obs, goal) - d(next_obs, goal)
- 需要信号: obs[0], obs[1], next_obs[0], next_obs[1]
- 本轮建议: 推荐作为 v1 主信号：奖励每一步更接近目标。
- 风险: 目标附近震荡。
- 后续迭代: 可 clip；后续配合成功、时间或稳定信号。

### distance_reward
- 角色: 密集过程引导
- 数学形态: -d(obs, goal)
- 需要信号: obs[0], obs[1]
- 本轮建议: 可作为小权重 anchor；不要和 progress_delta_reward 同时大权重堆叠。
- 风险: 接近目标但不完成；不关心速度和姿态。
- 后续迭代: 训练后检查 high_reward_without_success。

### stability_penalty
- 角色: 轻量稳定约束
- 数学形态: -lambda_v*|velocity| - lambda_a*|angle| - lambda_w*|angular_velocity|
- 需要信号: next_obs[2], next_obs[3], next_obs[4], next_obs[5]
- 本轮建议: 如果任务要求稳定接近/着陆，v1 可以小权重加入。
- 风险: 过强会保守或不敢动。
- 后续迭代: 若高速撞击或姿态失稳，增大权重。

### soft_landing_proxy
- 角色: 任务完成近似信号
- 数学形态: small_bonus if near_target and low_speed and stable_angle and both_contact else 0
- 需要信号: position, velocity, angle, contact flags
- 本轮建议: 可选小权重；不能直接把 contact 当 success。
- 风险: 如果条件太宽，会变成 contact reward hacking。
- 后续迭代: 如果 high_reward_without_success，收紧条件或移除。

### terminal_success_reward
- 角色: 任务目标奖励
- 数学形态: R_success * I[success]
- 需要信号: 显式 success flag
- 本轮建议: 若 explicit_success_flag_available=false，不作为 v1 核心项。
- 风险: 会诱导 LLM 发明 info['success']。
- 后续迭代: 当 wrapper 明确暴露 success 后再加。

### terminal_failure_penalty
- 角色: 失败惩罚
- 数学形态: -R_failure * I[failure]
- 需要信号: 显式 failure flag 或 termination_reason
- 本轮建议: 若 explicit_failure_flag_available=false，不作为 v1 核心项。
- 风险: 误判终止原因。
- 后续迭代: 当能区分失败终止后再加。

### time_penalty
- 角色: 效率约束
- 数学形态: -lambda_time
- 需要信号: 每步调用
- 本轮建议: 通常后续加入，不建议 v1 太早加入。
- 风险: 可能导致冒险或快速失败。
- 后续迭代: 若能接近但拖太久，再小权重加入。

### energy_penalty
- 角色: 动作/能耗约束
- 数学形态: -lambda_action * engine_use(action)
- 需要信号: action
- 本轮建议: 通常后续加入，v1 太早加入可能不敢动。
- 风险: agent_afraid_to_move。
- 后续迭代: 能接近并稳定后再优化燃料。

### gated_reward
- 角色: 安全门控
- 数学形态: if unsafe then penalty else task_reward
- 需要信号: 明确 unsafe 条件
- 本轮建议: v1 不建议使用复杂门控。
- 风险: 门控过严导致学不到。
- 后续迭代: 若安全被进度奖励抵消，再加入。

### potential_based_shaping
- 角色: 势能塑形
- 数学形态: gamma*Phi(next_obs)-Phi(obs)
- 需要信号: 可定义 Phi
- 本轮建议: 不作为 v1 首选；比 progress_delta 更抽象。
- 风险: Phi 错误会误导学习。
- 后续迭代: 如果需要更标准的 shaping，再替换或补充。

## 3. reward_v1 生成要求
- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget，而不是固定组件数量。
- 推荐 2~4 个组件：1 个主学习信号 + 0~2 个稳定/安全约束 + 0~1 个任务完成 proxy。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty 作为 v1 核心项。
- 如果速度/姿态信号明确可用，且任务需要稳定接近或着陆，可以加入轻量 stability_penalty。
- 如果使用 contact，只能作为 soft_landing_proxy 的一部分，必须和 near_target、low_speed、stable_angle 组合，不要直接把 contact 当 success。
- energy_penalty、time_penalty、gated_reward 默认后续迭代再加入。
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。