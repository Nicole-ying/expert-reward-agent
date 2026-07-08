# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
控制一个位于画面顶部中心附近的 2D 飞行器，在随机初始推力作用下，**快速且稳定地降落到中央目标平台**，同时**尽量节省引擎燃料**。  
理想表现是：平稳接近目标、减速、保持小角度、双腿安全接触平台并让身体停稳。

---

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

---

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: (8,)  
- **dtype**: float32（推断）  

各维度含义（均为相对值）：
- **obs[0]** – `x_position`：飞行器质心相对目标平台中心的水平坐标（目标中心为 0）
- **obs[1]** – `y_position`：飞行器质心相对目标平台高度的垂直坐标（平台高度为 0）
- **obs[2]** – `x_velocity`：水平线速度
- **obs[3]** – `y_velocity`：竖直线速度
- **obs[4]** – `body_angle`：机身朝向角（弧度或度，通常 0 表示竖直向上）
- **obs[5]** – `angular_velocity`：角速度
- **obs[6]** – `left_support_contact`：左腿/左支撑板是否接触（1.0 接触，0.0 未接触）
- **obs[7]** – `right_support_contact`：右腿/右支撑板是否接触（1.0 接触，0.0 未接触）

---

## 4. 动作空间 action_space
- **type**: Discrete(4)  
- **动作 0** – `no_engine`：不点火，无推力  
- **动作 1** – `left_orientation_engine`：点燃左侧姿态调整引擎（产生转向力）  
- **动作 2** – `main_engine`：点燃主引擎（产生主推力，一般向上）  
- **动作 3** – `right_orientation_engine`：点燃右侧姿态调整引擎（与 action 1 反向）

---

## 5. step 与终止条件分析
### 5.1 终止模式
终止触发由布尔表达式决定：  
`terminated = crash_or_body_contact OR horizontal_position_outside_viewport OR body_not_awake_or_settled`

- **success-like termination**：  
  - `body_not_awake_or_settled`：机身稳定、停止运动（通常因双腿着地且速度/角速度足够小）。  
  - `crash_or_body_contact` 在降落成功时也会为真（双腿接触平台），两种条件可能同时满足。  
  → 成功可由 `terminated=True` 时的状态特征判断（接近原点、小速度、双腿接触、小角度）。

- **failure-like termination**：  
  - `horizontal_position_outside_viewport`：水平漂出视野，必然失败。  
  - `crash_or_body_contact` 在**非安全接触**（如高速撞击地面、侧翻、头部触地）时触发，属于失败。  
  → 失败状态一般表现出大速度、大角度、位置偏移、单腿或零腿接触。

- **ambiguous termination**：  
  - 无显式成功/失败标志，所有终止都需要通过 `obs` 内容区分。

- **truncation**：  
  - 未定义时间截断，目前返回的 `truncated` 为 `False`。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: **false**  
- **explicit_failure_flag_available**: **false**  
- **allowed_info_fields**: 无（`step` 返回 `{}`，禁止使用 info）  
- **forbidden_or_uncertain_info_fields**: 所有 info 字段不可用；原始奖励 `original_reward` 被屏蔽，不可访问。

---

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs`（终止前一步的观察，或任意步骤的当前观察）
- `action`（当前执行的动作）
- `next_obs`（执行动作后的观察）
- `info` 仅限空字典，无可用字段
- `training_progress` 仅在明确被允许时使用，此处不开放

**禁止使用**：
- `original_reward`（官方奖励已屏蔽）
- 任何未声明的 info 字段
- 未在观察含义中声明的 obs 切片
- 外部环境属性或时间信息

---

## 7. 可用于奖励函数的信号
基于观察与动作，可构造的奖励驱动信号包括：

- **位置**：`next_obs[0]`（水平偏差），`next_obs[1]`（垂直偏差） → 衡量接近目标中心  
- **速度**：`next_obs[2]`（水平速度），`next_obs[3]`（垂直速度） → 控制软着陆  
- **朝向**：`next_obs[4]`（角度） → 保持竖直  
- **角速度**：`next_obs[5]`（角速度） → 抑制旋转  
- **接触**：`next_obs[6]`（左腿接触），`next_obs[7]`（右腿接触） → 奖励双腿平稳着地  
- **动作/引擎使用**：`action` ID → 惩罚燃料消耗（主引擎与姿态引擎的代价可区分）

---

## 8. 不确定或不可用的信号
- **原始奖励**（`original_reward`）完全屏蔽，不可用。  
- **info 字典**为空，无任何终止原因、成功标志或接触类型信息。  
- **是否坠毁 / 安全着陆** 无直接 flag，必须用观察（速度、角度、位置、接触）间接推断，且可能存在边界模糊情况。  
- **任务时间 / 步数** 未暴露，不能直接用于奖励中的时间惩罚。



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
- best_score_so_far: -108.900

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| progress_delta + soft_landing_proxy + stability_penalty | 4 | -108.900 | -373.710 | unsolved |

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
