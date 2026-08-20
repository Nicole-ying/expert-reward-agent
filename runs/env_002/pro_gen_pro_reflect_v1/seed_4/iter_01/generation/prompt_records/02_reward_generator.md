# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境事实、任务画像、奖励职责拆解、职责-信号映射；
2. expert_reward_context.md：固定专家 Schema，包括任务类型示例和 Formula Operator Library；
3. optional masked_step_source：默认不提供，除非调试开启。

你的任务不是机械选择某个 skeleton，而是：
1. 读取 environment_card.md 的 `expert_task_profile`；
2. 读取 `reward_role_decomposition`，明确 mandatory / conditional / avoid roles；
3. 使用 `role_to_signal_mapping` 检查每个职责可用的 obs/action/info 信号；
4. 从 expert_reward_context.md 的 Formula Operator Library 中为每个 selected role 选择数学形式；
5. 生成第一版奖励函数 `reward_v1.py`，并附带简短设计说明。

# Expert Schema 使用规则

- environment_card.md 中的 `reward_role_decomposition` 优先级高于 expert_reward_context.md 的模板。
- expert_reward_context.md 只提供专家模板和公式算子，不是固定答案。
- 先选 role，再选 signal，再选 formula operator，最后才写代码。
- 如果某个 role 没有可用信号，必须放入 excluded_roles，不得硬写。
- 如果 task_profile 与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不允许因为模板里提到某个 role 就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康/安全约束；效率、能耗、复杂门控和动态权重默认后续迭代再加入。

# 总体设计原则

- 从简单到复杂，但”简单”不等于只有一个组件。
- 不要用”最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 写完 reward 后自检：① 每个终止条件是否有前兆软信号？② 任务目标是否有直接的进度信号？③ 动作维度 ≥ 6 时，是否缺少效率约束（即使权重很小）？
- 不要机械照抄 expert template 或 formula operator。
- 不要使用 original_reward。
- 不要计算 fitness_score 或 fitness_score components。
- 不要使用未声明的 info 字段，例如 info["success"]、info.get("success")。
- 不要使用未声明的 obs 切片，例如 obs[0:3]。
- 只能使用 environment_card.md 声明的观测维度和索引，不得自行扩展为未声明的二维、三维或其他结构。
- 如果 explicit_success_flag_available=false，不要把 terminal_success_reward 写成 v1 核心项。
- 如果 explicit_failure_flag_available=false，不要把 terminal_failure_penalty 写成 v1 核心项。
- 允许使用 obs 和 next_obs 的逐 index 变量。
- 尽量让奖励平滑；需要距离、速度等连续项时，优先使用连续函数。
- 如果需要 sqrt，禁止 import numpy，使用 `** 0.5`。
- 如果想使用 exp 形式的平滑变换，禁止 import numpy；可以使用 `2.718281828 ** (...)`，并显式写 temperature 参数。

# 任务无关设计原则

## 原则 1：信号可用性优先

- 先检查 environment_card.md 中声明的可用信号、禁止信号和 role_to_signal_mapping。
- 只有当信号确实存在于环境接口中时，才设计依赖该信号的组件。
- 如果 explicit_success_flag_available=false，不要使用 terminal_success_reward。
- 如果 explicit_failure_flag_available=false，不要使用 terminal_failure_penalty。
- 不要发明未声明的 info 字段或 obs 切片。

## 原则 2：稠密性

- 优先选择每步都能提供有意义梯度的连续信号。
- 二值条件信号触发率过低时等于摆设。
- 连续函数、bounded 函数、soft proxy 通常比硬阈值更利于学习。

## 原则 3：尺度与平衡

- 不同组件的量级应大致可比，不要让一个组件在数值上统治其他组件。
- 约束/惩罚不应无条件压制任务驱动力；具体尺度必须结合触发频率、数学形态和预期行为判断。
- 差分信号、持续状态奖励和稀疏事件奖励具有不同时间语义，不能仅凭步均值比例判断谁更重要。

## 原则 4：信号冲突

- 不要同时大权重使用两个计算同一物理量的信号。
- 不要让惩罚项压制探索；过严姿态/速度/动作约束可能导致 agent 不敢行动。
- soft_health_gate 比强全局惩罚更适合处理“前进但失稳”的早期问题。

## 原则 5：阶段条件

- v1 阶段避免过早引入效率/动作代价；agent 应先学会任务方向，再优化效率。
- 复杂门控、动态课程、强能耗项默认后续迭代再加入。
- curriculum_weighting 只有当 training_progress 明确允许且任务确有阶段性冲突时才使用。

## 原则 6：可利用风险

- 每个组件都要考虑 agent 可能找到的捷径。
- 只奖励速度可能导致 velocity_burst_then_fall。
- 只奖励存活可能导致 stand_still 或 hover。
- 只奖励接触可能诱导 contact reward hacking。
- 直接奖励 vertical activity 可能诱导原地弹跳。

# role-based component budget

v1 推荐使用 2~4 个组件，按以下角色组织。专家模板和公式算子只提供设计启发，不限制你组合、变形或创造适合当前环境的新信号。

## 必须包含

**1 个主学习信号。** 这是 reward 的核心驱动力，告诉 agent “做什么能得分”。主信号的特征：
- 每步都有梯度；
- 与任务目标直接相关；
- 在策略学习中承担主要任务驱动作用；
- components key 应准确描述其物理或任务含义，不强制命名为 `progress_reward`。

## 允许包含（按需，不是必须全加）

- **0~2 个稳定/安全/健康约束。** 如果任务需要控制速度、姿态、身体高度、角速度等，可以加入轻量惩罚或 soft gate。约束的角色是“方向盘”而非“刹车”。
- **0~1 个任务完成近似信号。** 如果环境没有显式 success flag 但需要在 agent 接近完成时给予额外引导，可以用多条件组合的 soft proxy。proxy 必须由多个连续条件组合，不能直接伪造 success flag。
- **0~1 个效率/动作代价。** v1 默认不加或极小权重；能耗优化通常留到后续迭代。

## 默认不在 v1 使用

- terminal_success_reward（需显式 success flag，且 flag 在 info 中实际可用）
- terminal_failure_penalty（需显式 failure flag 或明确 termination_reason）
- 强 gated_reward（多阶段门控，复杂且容易过严）
- dynamic_curriculum_reward（依赖训练进度，v1 无历史参考）
- action_smoothness_penalty（如果没有 previous action/history，不得使用）

# 输出格式要求

函数签名必须完全一致：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

最终 reward 函数输出必须包含：
1. total_reward: float
2. components: dict，记录 individual reward components

首选返回格式：
```python
return float(total_reward), components
```

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
- 不要写 import。
- 不要写 class。
- 不要写 try/except。
- 不要写 eval/exec/open。
- 不要创建额外函数。
- 不要引入新的输入变量。
- 不要传 self；当前项目接口不是 Eureka 原版 self 接口。
- 不要使用 self attributes。
- 不要使用原始环境 reward。
- components 必须是 dict。
- components 只包含被加到 total_reward 的组件（A、B、C），不包含 total_reward 本身。

# Markdown 输出要求

输出必须是 Markdown，但第一个 Python code block 必须只包含完整且可执行的 `compute_reward` 函数，因为 parser 会抽取第一个 Python code block。

格式：

# reward_v1.py

```python
def compute_reward(...):
    ...
```

# reward_v1 设计说明

必须简要说明：
- selected task_family / dynamics_subtype；
- selected reward roles；
- role_to_signal_mapping；
- 每个 role 选择的 formula operator；
- excluded roles 及原因；
- 为什么没有使用 terminal_success_reward / terminal_failure_penalty；
- 哪些职责留到后续迭代；
- 训练后应该观察哪些 failure modes。

```

## User Prompt

```markdown
# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
环境要求一个双腿身体在不平坦地形上尽可能快、尽可能远地向前行走，同时最小化能量消耗。主体必须协调两侧髋、膝关节力矩，产生稳定的双足步态。**摔倒将导致回合提前终止**，因此保持身体平衡是隐含的刚性前提。次要是节省能量、获得平滑的关节动作，但它们绝不能以牺牲前进和平衡为代价。不应混淆“生存”与“到达终点”：虽然走过整个地形也算终止，但核心奖励必须源于持续的向前位移，而不是一次性的终点信号。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 任务核心是“持续向前通过（不平坦）地形”，主体是双腿形态，动作是连续的关节力矩。目标为延长前进距离和速度，并附带能耗/平滑要求，完全匹配 locomotion_continuous_control 族。无多个冲突严重的等价目标，因此不做 multi_objective_task。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（假定）
- obs[0]: hull_angle — 身体相对竖直方向的倾斜角度，reward_usable: true（防摔倒）
- obs[1]: hull_angular_velocity — 身体角速度，reward_usable: true（姿态稳定性）
- obs[2]: horizontal_velocity — 前进/后退线速度，reward_usable: true（**主前进信号**）
- obs[3]: vertical_velocity — 上下线速度，reward_usable: true（可辅助检测跳跃或不稳定）
- obs[4]: hip1_angle — 腿1髋关节角度，reward_usable: true（姿态约束）
- obs[5]: hip1_speed — 腿1髋关节角速度，reward_usable: true（关节平滑性）
- obs[6]: knee1_angle — 腿1膝关节角度，reward_usable: true（姿态约束）
- obs[7]: knee1_speed — 腿1膝关节角速度，reward_usable: true（关节平滑性）
- obs[8]: leg1_contact — 腿1着地标志，1.0=接触，0.0=无接触，reward_usable: true（步态节律/接触）
- obs[9]: hip2_angle — 腿2髋关节角度，reward_usable: true（对称/姿态）
- obs[10]: hip2_speed — 腿2髋关节角速度，reward_usable: true（平滑）
- obs[11]: knee2_angle — 腿2膝关节角度，reward_usable: true（姿态）
- obs[12]: knee2_speed — 腿2膝关节角速度，reward_usable: true（平滑）
- obs[13]: leg2_contact — 腿2着地标志，reward_usable: true（步态节律/接触）
- obs[14..23]: lidar_0..9 — 10 路激光测距值（前方地形高度轮廓），reward_usable: 不确定，通常不直接用于简单前进奖励，可预留为高级地形感知，但基础奖励设计中不建议使用，防止引入不必要复杂度。

所有观测均可用，但是否用于奖励需依据职责。

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- 范围: 每维 [-1.0, 1.0]
- action_dim 0: hip_torque_leg1 — 腿1髋关节力矩
- action_dim 1: knee_torque_leg1 — 腿1膝关节力矩
- action_dim 2: hip_torque_leg2 — 腿2髋关节力矩
- action_dim 3: knee_torque_leg2 — 腿2膝关节力矩

说明：动作是连续力矩控制，无离散选项。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（走完全程）
- failure-like termination: body_fallen_over（身体摔倒）
- ambiguous termination: 无
- truncation: 未在描述中提及，假定有最大步数限制（通用环境），但不作为奖励信号依据。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空，无法直接获得 reached_end_of_terrain 标志）
- explicit_failure_flag_available: false（同样无法直接获得摔倒标志）
- allowed_info_fields: 无（当前 step 返回 info={}）
- forbidden_or_uncertain_info_fields: 所有的 info 字段均不存在，因此不能依赖任何外部成功/失败标记。奖励函数只能基于观测和动作构建生存和前进信号。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前观测数组（24维），全部维度都可读取，但应仅使用与职责对应的维度。
- action：当前动作数组（4维）。
- next_obs：下一时刻观测数组（24维），用于计算变化量（如速度、姿态变化）。
- info：当前环境返回的 info 字典（本环境为空，不允许使用）。
- training_progress：仅在 prompt 明确允许时才使用，本描述未要求使用，因此禁用。

禁止使用：
- original_reward：被屏蔽的官方奖励，严禁依赖。
- 未声明的 info 字段（无可用字段，因此 info 完全禁用）。
- 未声明的 obs 切片（虽然已声明全部，但 lidar 不推荐用作基础奖励）。
- 任何隐式的成功或失败标记（如`done`标志），因为接口中不提供。

## 7. 可用于奖励函数的信号
- position: 无直接位置观测，但可通过积分水平速度推算（但由于基础，不宜复杂，建议仅用瞬时速度）
- velocity: obs[2] horizontal_velocity（主信号），obs[3] vertical_velocity（辅助），obs[1] hull_angular_velocity（姿态稳定）
- orientation: obs[0] hull_angle（倾斜惩罚）
- contact: obs[8] leg1_contact, obs[13] leg2_contact（步态节律/双脚支撑）
- action/engine: action 四个维度的幅值（能耗惩罚），action 的前后变化量（平滑惩罚）
- other: lidar 测距阵列（obs[14:23]），不建议用于基础奖励；关节角度/角速度可用于姿态正则化。

## 8. 不确定或不可用的信号
- 终点成功标志：无直接信号，无法奖励“到达终点”。
- 水平位置：没有任何位置坐标，无法计算累积位移，只能依赖瞬时水平速度。
- 着地力的大小：只有二值接触标志（1.0/0.0），不能得出冲击力。
- 地形信息：lidar 存在，但地形崎岖程度难以量化到标量奖励中，容易引入噪声。
- 外部终止原因：无法区分摔倒与到达终点，因此不能根据回合结束给予稀疏奖励。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bilateral_two_legs
  actuator_type: torque_motors_at_hip_and_knee
  contact_structure: bilateral_foot_ground_contact
primary_objectives:
  - maximize_forward_progress：最大化前向速度/位移
  - maintain_upright_balance：保持身体竖直（低倾斜角与角速度）
secondary_objectives:
  - minimize_energy：关节力矩尽量小
  - smooth_control：相邻动作变化小，减少震颤
main_failure_risks:
  - 身体倾斜过大、角速度发散造成摔倒
  - 步态不协调导致原地打转或后退
  - 过度追求低能耗而丧失前进动力
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 鼓励持续前进，是核心优化方向。
  why_required: 任务明确要求“走得尽量远、尽量快”，没有该职责将导致智能体呆立或摔倒。
  usable_signals: [obs[2] horizontal_velocity, obs[3] vertical_velocity（可选抑制跳跃）]
  risks: 仅使用瞬时速度可能鼓励快速而忽略平衡，需配合生存职责。

- role_id: upright_balance_survival
  purpose: 防止摔倒，保持身体竖直。
  why_required: 摔倒即终止，影响学习效率；姿态稳定是有效前进的前提。
  usable_signals: [obs[0] hull_angle, obs[1] hull_angular_velocity]
  risks: 惩罚过重会惧动，导致智能体不敢迈步。

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency
  condition_to_use: 当智能体已能稳定前进一段时间（如训练中后期）或前进速度已较高时逐步引入，权重由低升高。
  usable_signals: [action 各维度的二次型或绝对值]
  risks: 如果过早或过重引入，会抑制探索，使智能体选择不动作以最小化惩罚。

- role_id: action_smoothness
  condition_to_use: 当观察到关节震颤、高频抖动，或关节速度变化剧烈时加入。
  usable_signals: [action_t - action_{t-1} 的范数]（需要在奖励函数内部缓存上一个动作）。
  risks: 过度平滑会降低响应速度，阻碍快速迈步。

- role_id: contact_gait_rhythm
  condition_to_use: 当步态明显异常（例如双脚长期同时离地，或长期单脚支撑导致旋转）时尝试加入，用于引导交替支撑模式。
  usable_signals: [obs[8] leg1_contact, obs[13] leg2_contact]
  risks: 设计不当会奖励奇怪的接触序列；该环境非必须，可列为可选。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: terrain_awareness_from_lidar
  reason: lidar 数据维度高，抽取可用标量奖励困难；本环境的基础前进无需预知地形起伏，避免复杂化。
  forbidden_or_missing_signals: 无直接可用的地形崎岖度或障碍物标签。
  
- role_id: terminal_sparse_success
  reason: 缺少 reached_end_of_terrain 显式标志，无法可靠奖励终点。
  forbidden_or_missing_signals: info 为空，终止原因不可知。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | obs[2] horizontal_velocity, (可选)obs[3] vertical_velocity | 水平位置 | dense_state_signal, bounded_linear_reward, velocity_clip | 直接奖励正速度，可加竖直速度惩罚防跳跃 |
| upright_balance_survival | obs[0] hull_angle, obs[1] hull_angular_velocity | 无 | quadratic_penalty, bounded_signal (angle_limit) | 角度、角速度二次惩罚或设定生存阈值 |
| energy_efficiency | action[0..3] | 无 | quadratic_penalty, l2_norm, normalized_action_cost | 力矩平方和惩罚，系数逐步加大 |
| action_smoothness | (需要上一步动作缓存) action_diff | 无 | l2_norm(action_t - action_{t-1}) | 缓存动作向量，使用时注意初始化 |
| contact_gait_rhythm | obs[8] leg1_contact, obs[13] leg2_contact | 多接触力细节 | binary_contact_pattern_reward | 例如交替接触奖励，或惩罚双脚同时离地 |
| terminal_sparse_success | (None) | 终止分类标志 | — | 本环境不可用，禁用 |
| terrain_awareness_from_lidar | obs[14:23] | 无地形高度解释器 | — | 复杂、易过拟合，禁用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 智能体不敢动或全零动作 | 回合极短，前进速度始终≈0，动作幅值极小 | 降低能量/平滑惩罚权重，确保前进奖励主导 |
| 快速摔倒（身体倾斜过大） | avg hull_angle 很大，或角速度均值高，episode length 小 | 增强 upright_balance 惩罚，或增加姿态角约束（如 >0.5 rad 严重惩罚） |
| 原地踏步、位移为零但动作不停 | horizontal_velocity 均值接近零但动作幅度不小 | 奖励只用正速度，可能需要增加“最小起身”或禁止后退；考虑使用接触节奏奖励促交替迈步 |
| 身体颤抖、动作高频震荡 | action 序列的高频分量明显，joint speed 波动大 | 引入 action_smoothness 职责，或对动作变化做惩罚 |
| 走一段后原地打转或后退 | horizontal_velocity 出现较大负值，hull_angle 周期性摆动 | 增加旋转/横向速度惩罚（若有对应观测



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


```
