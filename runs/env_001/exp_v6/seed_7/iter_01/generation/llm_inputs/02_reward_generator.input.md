# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
被匿名化的环境是一个 2D 类飞行器轨迹优化任务。  
主体从视口顶部中央附近出发，带有随机初始力。  
目标是在尽可能短的时间内到达并稳定降落在中央的目标着陆垫上，同时尽可能少地使用引擎推力。  
智能体需要学会靠近目标、减小速度、保持稳定的姿态并实现安全的接触着陆。

---

## 2. 任务类型选择
selected_route_id: multi_objective_task  
confidence: high  
reason: 任务描述明确要求平衡“到达速度”与“引擎使用最少”两个目标，属于典型的多目标优化问题（快速 vs 燃料节约）。

---

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float (通常为 float32)
- obs[0]: x_position — 横向坐标，相对于目标着陆垫中心的水平距离
- obs[1]: y_position — 竖直坐标，相对于着陆垫高度的垂直距离
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体朝向角度
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左侧支撑腿接触标志，1.0 接触，0.0 未接触
- obs[7]: right_support_contact — 右侧支撑腿接触标志，1.0 接触，0.0 未接触

---

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine — 不点火，保持惯性
- action 1: left_orientation_engine — 启动左姿态微调引擎
- action 2: main_engine — 启动主引擎（产生主推力）
- action 3: right_orientation_engine — 启动右姿态微调引擎

---

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**：环境中未提供显式成功标志。当触发 `body_not_awake_or_settled` 且智能体已稳定着陆在垫上（位置接近0、速度极小、两侧支撑腿接触）时，可视为成功。
- **failure-like termination**：
  - `crash_or_body_contact`：与非垫面或障碍物发生碰撞，或机体与地面猛烈接触，视为失败。
  - `horizontal_position_outside_viewport`：水平位置超出边界，视为失败。
- **ambiguous termination**：
  - `body_not_awake_or_settled` 本身可能是成功着陆后的稳定状态，也可能是在空中失去动力后掉落到无效区域后的静止。需结合观测（位置、接触）才能区分。
- **truncation**：step 返回的 truncated 值为 `False`，无时间限制截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （无，step 返回的 info 为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，尤其禁止假设 `info["success"]`、`info["failure"]` 或类似键存在。

---

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs：当前步的完整观测向量
- action：当前步执行的动作
- next_obs：执行动作后的观测向量
- training_progress：仅在环境 prompt 明确允许时使用（这里未声明，故不推荐依赖）

禁止使用：
- original_reward：已被掩码的官方奖励，严禁直接或间接使用
- info：当前为空字典，不可假设任何内容
- 未声明的 obs 切片或隐藏状态

---

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position)，next_obs[0], next_obs[1]
- velocity: obs[2] (x_velocity), obs[3] (y_velocity)，及其 next_obs 对应项
- orientation: obs[4] (body_angle), obs[5] (angular_velocity)，及其 next_obs 对应项
- contact: obs[6] (left_contact), obs[7] (right_contact)，及其 next_obs 对应项
- action/engine: action 取值（0/1/2/3），可用于估算推力或燃料消耗

---

## 8. 不确定或不可用的信号
- 官方奖励 original_reward（被完全掩码，不可用）
- info 字典中的所有字段（当前为空，无成功/失败/原因等信息）
- 任何未在观察空间注明的内部物理量（如引擎推力大小、燃料剩余等）
- 外部地图或绝对世界坐标（观察仅提供相对目标垫的信息）



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。

## 1. 任务路由摘要
- multi_objective_task：按该任务类型选择主学习信号，并先检查接口可用性。

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