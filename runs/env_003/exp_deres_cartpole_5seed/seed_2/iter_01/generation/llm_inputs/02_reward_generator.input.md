# environment_card.md

# Env_003 环境理解卡片

## 1. 任务目标
在一个有限长度的轨道上，通过向左或向右对移动底座施加恒定大小的水平推力，使底座上未经驱动的摆杆尽可能长时间保持直立，同时底座不能超出轨道边界。任务本质是生存任务，目标是最大化存活时间步数，避免因杆倾倒或底座出界而终止。

## 2. 任务类型选择
selected_route_id: survival_balance_task
confidence: high
reason: 任务要求控制底座以维持不稳定杆的平衡，同时约束底座位置范围，没有明确的目标点或成功状态，唯一正向指标是存活步数，符合生存平衡类任务特征。

## 3. 观察空间 observation_space
- type: Box
- shape: [4]
- dtype: float32
- obs[0]: 底座水平位置，范围 [-4.8, 4.8]，超出 ±2.4 即终止，故有效范围为 (-2.4, 2.4)
- obs[1]: 底座水平速度，无界（但受动力学约束）
- obs[2]: 杆与竖直方向的夹角（弧度），范围 [-0.41887903, 0.41887903]，但终止阈值为绝对值大于0.20943951
- obs[3]: 杆的角速度，无界

## 4. 动作空间 action_space
- type: Discrete，2 个动作
- action 0: 向轨道负方向施加固定推力（push_negative_direction）
- action 1: 向轨道正方向施加固定推力（push_positive_direction）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无（不存在成功终止条件）
- failure-like termination: 
  - 底座位置绝对值超过 2.4（出界）
  - 杆角度绝对值超过 0.20943951 弧度（倾倒）
- ambiguous termination: 无
- truncation: 达到 500 步时截断（不代表成功或失败，只是时间限制）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 字典为空，{}）
- forbidden_or_uncertain_info_fields: 所有虚构字段（如 “success”, “failure”, “termination_reason” 等）均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：4 维观测（底座位置、速度、杆角、角速度）
- action：当前时刻执行的动作（0 或 1）
- next_obs：下一时刻观测（4 维）
- info：当前为空字典 {}，无任何可用字段

禁止使用：
- original_reward：已强制屏蔽，不可使用、不可逆向工程
- training_progress：未明确授权，不可用
- 未声明的 info 字段
- 未声明的 obs 切片（仅上述 4 个元素可用）

## 7. 可用于奖励函数的信号
- 底座位置 (obs[0] / next_obs[0])
- 底座速度 (obs[1] / next_obs[1])
- 杆角度 (obs[2] / next_obs[2])，可计算距直立的偏差
- 杆角速度 (obs[3] / next_obs[3])
- 动作 (action)，可用于惩罚频繁换向等
- 存活步数隐含在 termination/truncation 中，但不可直接用原始奖励，可通过自定义奖励鼓励存活（如每一步给+1）

## 8. 不确定或不可用的信号
- original_reward：完全屏蔽，禁止使用
- info 内任何字段：info 恒为空，故所有假定字段均不存
- 终止原因细节：环境只返回 terminated 布尔值，无法从 info 中获取是出界还是倾倒，不能直接作为奖励函数输入源，只能通过 obs/next_obs 特征自行推断



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架由任务路由检索生成，不预设特定组合。具体选择由环境接口中可用信号决定。

## 1. 任务路由摘要
- survival_balance_task：按该任务类型选择信号，并先检查接口可用性。

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