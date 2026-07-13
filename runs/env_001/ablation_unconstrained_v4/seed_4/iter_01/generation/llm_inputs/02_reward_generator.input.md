# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一个2D平面内的着陆器轨迹优化任务。智能体初始位于视口顶部中央区域，带有随机初始力。核心目标是**到达并稳定地停泊在画面中央的目标垫板上**，同时尽可能减少引擎推力总消耗，并且在接触垫板时保持低速度和接近直立的姿态。次要目标包括：减少燃料使用、缩短完成任务时间，以及避免任何不安全接触。该任务**不是**持续的行走推进，也不是无明确到达目标的平衡存活，也不是多目标权重均等的复合任务；一切奖励设计最终服务于 “到达、减速、稳定着陆” 这一中心目的。

## 2. 任务类型选择
- selected_route_id: navigation_goal_reaching  
- confidence: high  
- reason: 任务的核心成功条件是到达并稳定在指定的目标垫板上，附属要求（节能、快速、姿态稳定）均为服务于一次成功到达的优化目标，不具有与到达目标并列的权重。符合导航目标到达类任务的特征，但不同于单纯导航，它强调到达后的“软接触”与稳定停泊。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（各维度均为浮点，接触标志以 0.0 或 1.0 的浮点形式出现）  
- 各维度含义与奖励可用性：  
  - obs[0]: x_position（相对目标垫板的水平坐标），reward_usable: true  
  - obs[1]: y_position（相对垫板高度的垂直坐标），reward_usable: true  
  - obs[2]: x_velocity（水平线速度），reward_usable: true  
  - obs[3]: y_velocity（垂直线速度），reward_usable: true  
  - obs[4]: body_angle（机体姿态角），reward_usable: true  
  - obs[5]: angular_velocity（角速度），reward_usable: true  
  - obs[6]: left_support_contact（左支撑点接触标志，1.0=已接触），reward_usable: true  
  - obs[7]: right_support_contact（右支撑点接触标志，1.0=已接触），reward_usable: true  

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 各动作含义（均为离散动作，不涉及连续幅度）：  
  - action 0: no_engine（不点火任何引擎）  
  - action 1: left_orientation_engine（点燃左侧定向引擎，主要用于顺时针或逆时针方向调整）  
  - action 2: main_engine（点燃主引擎，产生沿机体轴向的推力）  
  - action 3: right_orientation_engine（点燃右侧定向引擎，作用与左侧引擎相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  身体稳定停泊在目标垫板上（body_not_awake_or_settled 触发），且未发生 crash 或出界。  
- failure-like termination:  
  crash_or_body_contact（与地面或非目标区域的硬接触）、horizontal_position_outside_viewport（水平位置超出视口边界）。  
- ambiguous termination:  
  无特别模糊的情况，但**单独**依靠 body_not_awake_or_settled 无法区分成功与因卡死在错误位置而“静止”，需要结合位置和接触信号共同判断；另外在训练早期，可能因初始随机力漂移导致未能接触垫板而直接出界。  
- truncation:  
  未在描述中明确提及最大步数截断，但环境可能在超时后终止，具体截断逻辑被遮蔽，不应在奖励函数中隐式利用。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: info 为固定空字典 `{}`，**没有任何可用字段**。  
- forbidden_or_uncertain_info_fields: 官方 success/failure 标志、终止原因字符串等均不可用，因为步函数返回 info 为空。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 上一时刻的完整8维观察  
- action: 刚执行的动作（0~3的离散动作）  
- next_obs: 新时刻的完整8维观察  
- info: 只能安全地视为空字典，**不应从中读取任何值**  
- training_progress: 默认0.0，除非上层明确允许且需要，否则**避免使用**

禁止使用：
- original_reward: 该参数被遮蔽，不得作为信号原始值或基线  
- 任何 official_reward 的变体  
- 未在本卡片中声明的 info 字段（实际上全部禁止）  
- 任何假设的“成功”或“失败”标签  
- 对观察空间未定义的维度进行切片

## 7. 可用于奖励函数的信号
- position: 可通过 obs[0], obs[1] 及 next_obs[0], next_obs[1] 计算相对于目标的距离或位置变化。  
- velocity: 可通过 obs[2], obs[3] 与 next_obs[2], next_obs[3] 得到合速度大小或各分量。  
- orientation: obs[4], next_obs[4] 提供机体角度；obs[5], next_obs[5] 提供角速度。  
- contact: obs[6], obs[7] 与 next_obs[6], next_obs[7] 为两腿接触标志，可用于判定着陆状态。  
- action/engine: 动作序号可用于判断是否点燃主引擎（action==2）或定向引擎（action==1 或 3），从而计算燃料使用。  
- other: 由位置、速度、角度衍生出距离变化率（接近速度）、姿态变化等合成信号。

## 8. 不确定或不可用的信号
- 任何与“成功”或“失败”直接相关的标签（info 为空）。  
- 原环境奖励函数的具体数值或规律。  
- 环境内部物理参数（如质量、重力加速度、垫板大小）—— 这些属于先验知识且未被提供，只能从观察中隐式感知。  
- 时间步计数或总步数上限（除非上层在调用时作为额外参数显式传入，但不在标准接口内）。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with thrusters
  actuator_type: one main bidirectional thruster + two orientation thrusters (left/right)
  contact_structure: two support legs with binary ground contact sensors
primary_objectives:
  - 使机身质心到达并保持在目标垫板水平范围内
  - 着陆时垂直接近速度足够低（避免硬碰撞）
  - 着陆后机身姿态保持接近竖直（角度、角速度小）
secondary_objectives:
  - 最小化引擎点火频率/总推力时间
  - 在保证安全前提下尽快完成着陆
main_failure_risks:
  - 直接坠毁在垫板以外的地形上
  - 超出水平视口边界而强制终止
  - 着陆姿态严重倾斜导致单腿悬空，虽“接触”但不稳定
  - 燃料不足导致无法减速而高速撞击（虽然在观察中不直接暴露燃料量，但从动作频率可部分推断）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: proximity_to_target  
  purpose: 鼓励质心不断靠近垫板中心（xy 距离减小），是达成“到达”目标的最直接推动力。  
  why_required: 没有位置激励，智能体将无法学会朝目标移动。  
  usable_signals: [obs[0], obs[1], next_obs[0], next_obs[1]]  
  risks: 仅用终点距离会使智能体不会减速，可能高速撞上垫板被判定 crash 或导致不稳定。

- role_id: soft_landing_velocity_penalty  
  purpose: 在接近地表（小 y 距离）且已至少有一只脚接触时，严惩高速度（尤其是垂直分量）。  
  why_required: 确保着陆满足“安全接触”的要求，是区分成功与失败的关键。  
  usable_signals: [obs[1], next_obs[1], obs[2:4], next_obs[2:4], contact signals]  
  risks: 如果全局启用可能阻碍早期快速下降，需要条件化启用。

- role_id: upright_stability_penalty  
  purpose: 惩罚过大的机体倾斜角度和角速度，尤其是在接触阶段。  
  why_required: 着陆器倾覆是典型失败模式，必须由奖励直接压制。  
  usable_signals: [obs[4], obs[5], next_obs[4], next_obs[5]]  
  risks: 过于敏感可能阻止必要的姿态调整。

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency_penalty  
  purpose: 在不需要大幅调整时，惩罚不必要的引擎点火，以节省燃料。  
  condition_to_use: 当智能体已处于接近目标且速度足够低时，或不处在迫降阶段时启用。  
  usable_signals: [action, position, velocity signals]  
  risks: 过早使用会抑制学习探索动作导致无法到达目标。

- role_id: stable_stand_bonus  
  purpose: 当两腿均稳定接触且姿态竖直且速度接近零时，给予正向奖励推动“停稳”。  
  condition_to_use: 仅在明确检测到着陆成功倾向时启用，用于避免反复弹跳。  
  usable_signals: [next_obs[4:8], next_obs[2:4]]  
  risks: 如果触发条件过于宽松，可能奖励不完整着陆姿态。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_success_shaping  
  reason: 由于环境没有提供显式的 success 标志，无法从 info 中获得官方成功信号，强行通过硬编码边界（如“位置在垫板内且速度小且双触”即视为成功）会导致与真实物理终止不一致，甚至奖励假阳性。  
  forbidden_or_missing_signals: [explicit_success_flag]

- role_id: remaining_time_reward  
  reason: 未提供任务剩余步数或时间信息，环境也未声明时间奖励，避免引入不存在的进度赶时信号。  
  forbidden_or_missing_signals: [time_to_limit, internal_step_counter]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| proximity_to_target | next_obs[0:2] (x, y 距离)，obs[0:2] 用于增量 | – | distance_to_target (L2)， position_delta_reward (距离减小奖赏) | 注意 y 方向是相对垫板高度，y=0 代表机身已到达垫板高度线，此时距离仅剩水平偏移。 |
| soft_landing_velocity_penalty | next_obs[2:4] 的合速度/垂直分量，contact any (obs[6] 或 obs[7])，y_position | – | conditional_quadratic_penalty， multiplicative_activation (由接近地表和接触程度调节因子) | 必须仅在满足“接近地表且接触”条件时生效，防止过早减速。 |
| upright_stability_penalty | next_obs[4] (角度)，next_obs[5] (角速度) | – | absolute_value 或 quadratic_penalty | 可考虑在接触后进一步放大权重。 |
| fuel_efficiency_penalty | action == 1,2,3 (引擎点火) | 燃料



# expert_reward_context.md

# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

不要反过来先套某个 skeleton 名称。模板只提供专家思考方式，不构成封闭候选集合。

---

## 1. Expert Schema 使用规则

- environment_card.md 中的 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

每个算子包含：数学形式、适用场景、触发证据、反模式。

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2` 或 `w * exp_form`
    凸化形式在 signal 较大时提供更强梯度。触发证据：episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小——说明 agent 满足于低水平稳态，需要凸化奖励来打破。
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
    hinge 只在超出安全区间时生效，避免在安全范围内持续惩罚正常波动。触发证据：约束组件的 active_rate≈100% 但 terminated 率仍然很高——说明"全时惩罚"没有给 agent 安全探索空间，它无论怎么调整都被罚。
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：线性正奖励可能导致慢速平台；凸化形式若权重过大可能诱导极端行为；hinge 的 threshold 设太宽则防护不足。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - 平滑压缩: `x / (1 + abs(x))`
  - 倒数衰减: `1 / (1 + k * abs(error))`
  - 线性衰减: `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。
- 触发证据：某个信号的 episode_sum_mean 出现极端值（远大于其他组件），说明无界形式被 exploit。
- 风险：threshold 过小会导致反馈饱和或无梯度。
- 反模式：不要用 bounded_signal 替代 hinge penalty——如果目标是"只在越界时惩罚"，用 dense_state_signal 的 hinge 形式，不要用 bounded 包围。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 触发证据：有明确的目标度量（如到目标的距离）且该度量在 episode 中单调递减时 agent 表现好。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。
- 反模式：不要对速度类信号用 improvement_delta——持续速度本身已经是"进步"，delta 会退化为噪声。

### 2.4 potential_based_shaping
- 适用职责：有明确 potential function 的任务塑形。
- 常见形式：`gamma * Phi(next_obs) - Phi(obs)`
- 使用条件：能够从环境信号定义合理的 Phi。
- 风险：错误 Phi 会误导策略；reward_v1 不默认使用，除非任务天然适合。

### 2.5 quadratic_penalty
- 适用职责：姿态误差、角速度、动作幅度、速度等轻量约束。
- 常见形式：`-w * error**2` 或 `-w * sum(action_i**2)`
- 使用条件：约束信号可观测，且不应压制主学习信号。
- 风险：权重过大会导致 agent_afraid_to_move 或 over_conservative_policy。
- 触发证据：某维度出现高频大幅波动或极端值，但没有触发终止——说明需要轻量抑制而非硬约束。
- 反模式：不要对"有明确安全边界"的信号用 quadratic_penalty（如身体高度必须在 0.2-1.0）。quadratic 从中心开始罚，会让 agent 困在中心不敢动；应改用 hinge 形式只在边界附近生效。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * gate_factor`，gate_factor 在身体状态恶化时从 1 平滑衰减到 0。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (signal - danger) / margin))`
- 使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
- 触发证据（关键）：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍然 >0——说明 agent 在"先冲后死"，需要 gate 在健康恶化时切断主奖励，而不是加一个独立惩罚。
- 风险：gate 太严格会抑制探索；gate 的衰减区间应设在"接近危险但尚未终止"的范围内。
- 反模式：不要用"加大独立惩罚系数"替代 gate。如果 terminated 是因为身体状态越界，单纯加大该状态的惩罚（Level 1）通常不如将其作为 gate 乘到主奖励上（Level 2），因为惩罚只在越界后才生效，gate 在越界前就开始衰减主信号。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 触发证据：agent 能在各个子条件上分别取得进展，但无法同时满足——说明缺一个"联合满足"的引导信号。
- 风险：乘积容易塌缩（一个 factor 趋近 0 则整体为 0）；使用 `(factor_1 + factor_2 + ...) / n` 或几何平均 `(factor_1 * factor_2 * ...) ** (1/n)` 可缓解。
- 反模式：不要用二值条件做乘积——每个 factor 必须是连续函数，否则乘积退化为稀疏信号。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

---

## 3. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"直接定位到"该选哪个算子做 Level 2 变换"。
不要求组件名完全匹配；以数学语义和训练表现证据为准。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

