# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
一个2D物体（类似可悬停飞行器）从画布顶部中心附近出发，受到一个随机初始力。  
任务目标是**尽快到达中央的目标平台并稳定停靠**，同时**尽可能少地使用引擎推力**。  
智能体需要学会靠近目标、减速、保持稳定姿态，并实现安全的接触着陆。

## 2. 任务类型选择
selected_route_id: multi_objective_task  
confidence: high  
reason: 任务明确要求同时优化多个目标——快速到达目标、最小化推力消耗、保持稳定方向、确保安全接触，属于典型的多目标权衡问题。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测，基于连续物理量）
- 各维度含义（按 index 顺序）：
  - obs[0]: x_position — 相对于目标平台中心的水平坐标
  - obs[1]: y_position — 相对于目标平台高度的垂直坐标
  - obs[2]: x_velocity — 水平线速度
  - obs[3]: y_velocity — 垂直线速度
  - obs[4]: body_angle — 机体方向角
  - obs[5]: angular_velocity — 角速度
  - obs[6]: left_support_contact — 左侧支撑接触标志（1.0 表示接触，0.0 表示未接触）
  - obs[7]: right_support_contact — 右侧支撑接触标志（1.0 表示接触，0.0 表示未接触）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作编号与含义：
  - action 0: no_engine（不点火，什么也不做）
  - action 1: left_orientation_engine（点燃左侧姿态引擎，产生右转力矩/推力分量）
  - action 2: main_engine（点燃主引擎，产生主要上升/减速推力）
  - action 3: right_orientation_engine（点燃右侧姿态引擎，产生左转力矩/推力分量）

## 5. step 与终止条件分析

### 5.1 终止模式
step 内部的终止信号由以下三个条件任意满足时触发（terminated = True）：
- **crash_or_body_contact**: 发生碰撞或与不应接触的表面接触 → **失败型终止**
- **horizontal_position_outside_viewport**: 水平坐标超出视野范围 → **失败型终止**
- **body_not_awake_or_settled**: 物体不再活跃或已稳定 → **模糊型终止**（可能是成功停靠，也可能是坠毁后静止或其他非理想静止）

未观察到显式的截断（truncation）信号（返回 `truncated=False`），因此本环境在分析时不考虑最大步数截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
  原环境 step 返回的 info 为空字典，不包含任何成功标志字段。
- explicit_failure_flag_available: false  
  同上，没有失败标志字段。
- allowed_info_fields: 无（info 为空，没有可用字段）
- forbidden_or_uncertain_info_fields: 所有可能存在的 info 字段均不可用（如 `success`, `failure`, `termination_reason` 等均不存在）。

> 注意：不能假设 info 中存在 `info["success"]` 或 `info["failure"]` 等字段用于奖励计算。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用的输入：**
- `obs`（当前观测）
- `action`（当前执行的动作）
- `next_obs`（下一观测）
- 以及 `info` 中被明确允许的字段（本环境中 `info` 为空，故不使用）

**严格禁止使用的输入：**
- `original_reward`（或任何形式的官方奖励）
- `info` 中未明确允许的任何字段（即全部字段）
- `training_progress`（除非 prompt 明确要求使用，本环境中未被要求）
- 任何对观测的未声明的隐含切片或解释（需基于已声明的字段含义使用）

## 7. 可用于奖励函数的信号
以下信号均可由 `obs` 和 `next_obs` 安全构造，并用于设计奖励分量（但奖励本身不由本卡片设计）：
- **位置偏差**：水平位置 `obs[0]`（或 `next_obs[0]`），垂直高度 `obs[1]`。
- **速度**：水平速度 `obs[2]`，垂直速度 `obs[3]`，可合成合速度大小。
- **姿态**：机体角度 `obs[4]`（偏离竖直或目标方向的角度）。
- **角速度**：`obs[5]`。
- **接触信号**：左接触 `obs[6]`，右接触 `obs[7]`；可用于判断是否着陆、是否双足平稳接触。
- **动作/发动机使用**：动作编号（0~3）本身，可用于衡量推力使用频率（如主引擎使用惩罚）。

衍生的复合信号举例（可不直接列出，此处仅为澄清）：
- 到目标点的欧氏距离：基于 obs[0], obs[1]。
- 速度大小、加速度变化（通过 obs 与 next_obs 速度差近似）。
- 稳定性指标：低速度 + 低角速度 + 双接触标志同时为 1 + 小角度偏差。

## 8. 不确定或不可用的信号
- **绝对成功/失败标志**：无法从 info 中获得，必须通过观测自构建判断逻辑。
- **目标平台的具体边界坐标**：仅知道“中心”和“平台高度”的参考系，但未给出容差范围，需要在奖励中保守假设或通过学习推断。
- **燃油存量或剩余能量**：观测中不包含该量，只能通过动作序列间接估计推力消耗，但不能获得精确能耗。
- **原环境内置奖励**：被屏蔽，严禁使用。
- **任何额外 info 字段**：如 None、未出现的键值均不可假设存在或可用。



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架由任务路由检索生成，不预设特定组合。具体选择由环境接口中可用信号决定。

## 1. 任务路由摘要
- multi_objective_task：按该任务类型选择信号，并先检查接口可用性。

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
- 从上述任务路由推荐的骨架中选择，优先选择所需信号在环境接口中可用的骨架。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。



# ⚠️ Restart Context

以下骨架在前序迭代中已尝试但未成功：
- progress_reward + soft_landing_proxy + stability_penalty

上述骨架在前序迭代中已尝试但未取得突破。
请基于训练证据选择改进方向：
- 如果认为同一骨架仍有可修复空间（如系数调节、条件化约束），可以继续在当前骨架上修改。
- 如果诊断表明当前骨架存在结构性问题（如信号冲突、梯度消失），请从 expert_reward_context.md 中选择不同的数学形态。
- 不要机械重复已失败的骨架。
