# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。先用训练证据解释失败，再选择最小且可验证的干预。你的目标不是匹配某个已知环境或骨架名称，而是改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- feedback来自训练后固定策略的同一批评估轨迹。`episode_sum_mean`表示每回合有符号累计量，`magnitude_share`表示绝对累计量份额，`signed_share`保留净方向，`active_rate`表示非零触发率（旧日志中的`nonzero_rate`）。
- 组件统计是观察证据，不是因果贡献。必须结合score、episode_length、terminated/truncated、历史修改及其结果判断。
- 不同时间语义不可直接比较：逐步差分、持续状态值、惩罚和稀疏事件bonus不能套同一个比例阈值。
- 不得仅因任务描述出现“跳跃、着陆、抓取”等语义，就断言对应状态量是缺失奖励职责。新增职责必须有轨迹行为、终止分布、组件激活或历史干预结果支持；证据不足时明确写“未知”，优先保持best主结构并提出最小可证伪修改。
- episode达到时间上限且失败终止很少时，首先判断现有主信号是否已经实现稳定行为、剩余差距是否来自效率或主目标强度；没有行为证据时，不为动作过程本身添加proxy。

# 唯一决策流程

按顺序完成，不能因知识库或工具命中某个变换就跳级。

## 1. 行为与历史诊断

拿到训练反馈后，必须先明确回答原有三个问题：

1. **这个agent发生了什么？** episode相对正常长度明显偏短，可能在快速失败；episode很长但得分差，可能在徘徊；得分已经好但某组件尺度异常，可能存在exploit，必须用行为证据验证。
2. **哪个组件最值得干预？** 结合组件数学形态、episode_sum_mean、signed_share、magnitude_share、active_rate、外部score和episode_length判断，不得把数值占比直接写成因果贡献。
3. **我之前改了什么？** 从Agent Memory检查上一轮动作、预测和实际效果。如果上次改了A但得分没有实质变化，这次不要再次修改A。

随后确认当前失败是否来自该组件或某个缺失职责，一次只选择一个干预目标。current明显差于best时，以best代码为基础，但必须做一个新的、有证据的修改，不能原样复制best。

## 2. 信号完备性检查

检查当前奖励是否具有任务所需且可达的职责，而不是要求固定组件名称：

- 任务进展或可学习的过程引导；
- 必要的稳定、安全或动作约束；
- 当过程最优不等于任务完成时，能区分联合满足或完成状态的信号。

如果必要职责缺失、active_rate接近0、数学形态使反馈塌缩，或proxy与外部任务明显错位，进入Level 2。若职责基本完备、符号与数学形态合理，只是相对尺度异常，先进入Level 1。

## 3. 选择干预层级

### Level 1：尺度修复

适用条件：组件职责、符号和数学形态合理，证据主要表明单个组件过强或过弱。

- 只调整一个组件的系数，其他组件保持不变。
- 对同为逐步稠密信号、且progress明确承担主引导职责的早期学习阶段，旧实验中的`ratio_to_progress`思想仍然适用：`|penalty/progress| > 0.5`可作为优先检查并尝试降系数的经验触发器；`penalty/progress ≈ 0.1`可作为轻约束起点。
- 上述比例是v6实验中的有效经验先验，不是因果结论或通用硬阈值；事件bonus、持续状态奖励和正负抵消严重的差分均值不能直接套用。
- 若一次明确的尺度修复后，尺度异常已经消失但外部行为没有实质改善，不继续反复调同一系数，转Level 2。

### Level 2：有方向的数学结构变换

适用条件：必要信号缺失/不可达，或证据直接否定当前数学形态，或Level 1已修复尺度但策略仍失败。每轮只改变一个目标组件；改变该组件形态时同步设置与新值域匹配的系数，仍算一次组件干预。

| 证据模式 | 结构变换 | 下一轮验证 |
|---|---|---|
| 任务事件几乎不触发，缺少局部反馈 | sparse_to_dense：稀疏事件→连续过程证据 | active_rate及外部表现改善，且不产生proxy徘徊 |
| 极端值支配奖励 | unbounded_to_bounded：无界→归一化有界 | 极端轨迹支配下降，得分方差下降 |
| 占据好状态即可持续获奖 | state_to_improvement：状态值→状态改善量/有效势能差 | 停留不再积累收益，任务进展改善 |
| 约束在无关阶段妨碍探索 | global_to_local_gate：全局→阶段相关/局部门控 | 早期探索与局部约束同时改善；先确认不是单纯尺度过强 |
| 独立目标可互相补偿 | independent_to_joint：加权和→联合满足 | 单项刷分减少，必要条件共同改善 |
| 多个小因子相乘导致塌缩 | product_to_noncollapsing_joint：乘积→几何平均/软最小/门控和 | 非零反馈增多且联合约束保留 |
| 持续事件被重复领取 | persistent_to_transition_event：持续状态→有效状态转移 | 重复积累下降，外部完成保持或改善 |
| proxy提高但外部任务不升 | proxy_to_completion_alignment：代理目标→任务完成对齐 | proxy与外部分数重新同向 |
| 复杂耦合无法诊断 | coupled_to_diagnostic_components：耦合→少量直接组件 | 组件可解释并形成单一干预假设 |
| 稠密proxy形成中等分平台 | dense_to_task_event：全程proxy→局部/转移任务信号 | 刷新best，完成相关行为增加 |

常用数学性质：二值稀疏条件信用分配困难；连续乘积表达联合满足但可能塌缩；加权和反馈稠密但允许目标补偿；bounded函数限制极端值但输入仍需按环境尺度归一化；门控只应在证据表明“作用阶段错误”时使用。

### Level 3：重建主骨架

满足任一条件时停止局部修补：

- 同一骨架家族已迭代2轮以上，且历史最佳得分仍未超过target的25%；
- 同一结构家族连续2轮以上未刷新best，且至少做过一次Level 2；
- Level 2改变数学形态后没有实质改善；
- 同一结构家族连续3轮未刷新best且仍未达到target，即使已超过target的25%也要警惕中等分平台。

Level 3可以更换主信号框架或重新组合少量组件。expert_reward_context中的骨架是设计原语和风险提示，不是封闭候选列表；可以采用、组合、变形或基于环境事实创建新结构。

# 工具

核心Level判断必须依靠本Prompt完成。仅在根因不确定、多个Level 2变换难以区分或需要骨架细节时调用一次最相关工具：

- `search_reward_design_knowledge(query)`：检索相似失败模式和经验修复。
- `get_skeleton_detail(skeleton_name)`：查看数学形态、原理和陷阱。
- `get_reward_transformation(query)`：查看结构变换的原理、风险和验证目标。

# 代码约束

- 禁止terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的obs、next_obs、action和info字段，不得发明字段、切片维度或新输入。
- 第一个Python code block只能包含一个完整的`compute_reward`函数；不要写import、class、try/except或额外函数，不要使用self。
- 禁止eval/exec/open，禁止使用original_reward或原始环境reward。
- 需要平方根时使用`** 0.5`，禁止import numpy。需要指数形式时使用`2.718281828 ** exponent`，或改用`1/(1+k*x)`、`max(0,1-x/D)`等无需库的bounded表达式。
- 除Level 3重建外，每轮只修改一个目标组件，不顺带调整其他组件。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回`(float(total_reward), components)`；components只放总公式中直接出现的奖励组件，不放total_reward和中间调制器。

# 输出

先用以下固定字段各写一句，不复述输入表格：

1. `evidence`：支持判断的外部结果、组件证据和上一轮结果；
2. `behavior_diagnosis`：策略当前的失败行为；
3. `signal_completeness`：必要职责是否完备、可达；
4. `selected_level`：Level 1、Level 2或Level 3及触发条件；
5. `selected_intervention`：唯一目标组件及具体修改；
6. `falsifiable_hypothesis`：为什么该修改应改善策略；
7. `expected_next_round`：下一轮哪些指标应如何变化；
8. `main_risk`：最可能引入的新漏洞。

然后立即输出完整Python代码。预期必须能被下一轮反馈证伪。

```

## User Prompt

```markdown
# Search objective
- target_score: 200.000000
- current_score: 231.371376
- gap_to_target: -31.371376
- target_achievement_ratio: 115.686%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 231.371376）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract next state signals
    x = next_obs[0]
    y = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distance to target platform center (goal is (0,0))
    distance = (x**2 + y**2) ** 0.5

    # ---------- main learning signal: goal proximity ----------
    w_prox = 2.0
    # Bounded positive reward: maximum 2.0 at distance 0, decays smoothly
    prox_reward = w_prox / (1.0 + distance)

    # ---------- near‑factor for soft‑landing constraints ----------
    near = 1.0 / (1.0 + distance)   # 1 when close, ~0 when far

    # ---------- velocity penalty ----------
    w_vel = 0.1
    vel_penalty = -w_vel * near * (x_vel**2 + y_vel**2)

    # ---------- body angle penalty ----------
    w_angle = 0.2
    angle_penalty = -w_angle * near * (angle**2)

    # ---------- angular velocity damping ----------
    w_angvel = 0.1
    angvel_penalty = -w_angvel * near * (ang_vel**2)

    # ---------- contact bonus (both feet on ground) ----------
    w_contact = 1.0
    both_contacts = left_contact * right_contact   # 0 or 1
    contact_reward = w_contact * near * both_contacts

    # ---------- total reward ----------
    total_reward = prox_reward + vel_penalty + angle_penalty + angvel_penalty + contact_reward

    components = {
        "proximity_reward": prox_reward,
        "velocity_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "angular_velocity_penalty": angvel_penalty,
        "contact_bonus": contact_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=231.371376, len=589.350000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[137.422777, 292.936248]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 1036.567659 | 76.8% | 76.8% | 100.0% |
| contact_bonus | 310.228859 | 23.0% | 23.0% | 53.3% |
| velocity_penalty | -2.417221 | -0.2% | 0.2% | 98.5% |
| angle_penalty | -0.422520 | -0.0% | 0.0% | 100.0% |
| angular_velocity_penalty | -0.186958 | -0.0% | 0.0% | 83.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个二维飞行器着陆任务。飞行器从视口顶部中央附近以随机初速释放，目标是**尽快且节能地到达中央目标平台并稳定停靠**。具体要求包括：精确水平定位到平台上方，垂直速度接近零，保持姿态平稳，并让左右支撑足安全接触地面。飞行动力受离散引擎推力、姿态控制和物理约束。主要目标是安全着陆，次要目标是节省推力、快速完成。不应该追求极速而忽略平稳接触或倾角。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推断)
- obs[0]: x_position，相对目标平台中心的水平偏移，reward_usable: true
- obs[1]: y_position，相对平台高度的垂直偏移，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体倾角，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左侧支撑足接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右侧支撑足接触标志（0/1），reward_usable: true

**注意**：所有8维均明确可用，reward_usable 为 true。但接触标志仅代表物理接地，并不代表成功停止，不能直接用作成功条件。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，无推力，仅靠惯性/重力。
- action 1: left_orientation_engine，启动左定向引擎（调节姿态）。
- action 2: main_engine，启动主引擎（提供主要推力，推测方向为机体下方或后方）。
- action 3: right_orientation_engine，启动右定向引擎。

**注意**：离散动作无幅度参数，每个动作是瞬时脉冲式发动机点火，持续时间由步长决定。不允许动作幅度或连续调节。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机体停止并可能处于稳定接触状态，可能暗示成功着陆）——这是最可能的成功信号，但需结合接触和位置判断。
- failure-like termination: crash_or_body_contact（碰撞或机体与地面非支撑足接触），horizontal_position_outside_viewport（飞出视口水平边界）。
- ambiguous termination: body_not_awake_or_settled 可能因各种原因（悬挂、静止于非目标位置）发生，本身不保证成功着陆。
- truncation: 无显式最大步数，源中 truncation 恒为 False（见返回 `terminated, False, {}`），环境不设置时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，非平台提供的 success 字段）
- explicit_failure_flag_available: false
- allowed_info_fields: 无，info 恒等于空字典 {}，不可用于奖励。
- forbidden_or_uncertain_info_fields: 所有 info 字段，因为不存在。

**注意**：不能依赖 info["success"] 等，必须通过观测构建成功判断。

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4])
- angular_velocity: obs[5]
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: 当前 action（离散值 0-3）可反映推力使用情况
- other: 可构造 derived 信号，如距离目标平台的距离 = sqrt(x^2 + y^2)，水平偏角等。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body (lander-like)
  actuator_type: discrete impulse engines (main + two orientation thrusters)
  contact_structure: two support feet (left/right) with binary contact detection
primary_objectives:
  - 到达目标平台正上方 (x≈0, y≈0)
  - 稳定停靠 (速度≈0, 倾角≈0, 双脚接触)
secondary_objectives:
  - 最小化发动机使用频率或总推力脉冲
  - 尽快到达 (速度适中，而非极速)
main_failure_risks:
  - 高速撞击目标平台，导致 crash_or_body_contact
  - 姿态失控翻滚导致接触地面或飞出视口
  - 过度使用推力导致高速或远离目标
  - 悬停过久耗尽推力后未能稳定接触
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity_and_arrival
  purpose: 推进机体向目标平台中心靠拢，并最终到达零位置。
  why_required: 环境核心任务是导航到达，无此职责无法引导智能体向目标移动。
  usable_signals: [x_position, y_position, derived distance_to_pad]
  risks: 若只奖励接近，可能鼓励以高速撞击而不减速，必须与软着陆职责配合。

- role_id: soft_landing_and_stabilization
  purpose: 在接近目标时强制减速、保持姿态水平、实现双脚平稳接触。
  why_required: 成功着陆要求低速和双脚接触，单纯到达可能忽略稳定。
  usable_signals: [x_velocity, y_velocity, body_angle, angular_velocity, left_contact, right_contact]
  risks: 过早施加稳定压力可能导致贪婪行为，不愿离开初始位置。需要阶段门控：仅在靠近目标且速度较高时施加减速惩罚，或在接触不完整时惩罚。

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency
  condition_to_use: 当智能体已具备基本到达能力后，可作为辅助项，鼓励减少非必要推力使用。  
  usable_signals: [action (0=无推力, 1/2/3=使用发动机)]  
  risks: 若从一开始就过度强调，可能阻碍学习起飞或调整动作。应渐序引入或使用小系数。

- role_id: terminal_settlement_bonus
  condition_to_use: 在终止条件 `body_not_awake_or_settled` 且双脚接触、位置接近零时提供一次性正向反馈，明确成功。  
  usable_signals: [terminated flag from env + obs contacts and positions]  
  risks: 无法在奖励函数内部直接获取终止标志（因为 compute_reward 在每步调用，且不一定知道终止），需从当前步的 next_obs 预测是否可能已经满足条件，或依赖后续环境返回的 done 但 compute_reward 没有 done 参数。实际上，在标准 reward 接口中，done 通常不可用，除非我们定义 reward 为 per-step 并基于 next_obs 推断。因此这个职责可能对环境接口不可实现，应列为 avoid 或在外部利用 done 时由 env 提供，但当前不允许。目前 env info 为空，缺乏终止信息，故**不推荐**在 compute_reward 内部实现。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: terminal_settlement_bonus
  reason: 终止条件信息不在函数签名中（无 done 参数），无法可靠判断是否达到终止。若仅基于 next_obs 猜测“可能成功”，有可能错误奖励非终止步。当前环境在 compute_reward 阶段不应引入任意成功判断。
  forbidden_or_missing_signals: [done flag, info.success]

- role_id: angular_velocity_penalty_direct
  reason: 虽然可用 angular_velocity，但仅关注角速度而忽略绝对角度可能导致机体回旋时受惩罚，而倾斜但静止不罚。应优先使用 body_angle 稳定，angular_velocity 仅作为阻尼项，不易单独作为职责。

- role_id: pure_time_penalty
  reason: 无显式步数截断，施加每步负奖励可能导致策略倾向于尽快结束 episode（包括通过失败终止），安全性下降。慎用，除非经过训练后期校验。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity_and_arrival | x_position, y_position (obs[0], obs[1]) | 无 | dense_state_signal, bounded_signal (e.g. negative L2 distance) | 可使用高斯、倒数或线性分段奖励形状，避免远距离奖励过大。 |
| soft_landing_and_stabilization | x_velocity, y_velocity, body_angle, angular_velocity, left_contact, right_contact (obs[2:8]) | 无 | quadratic_penalty (on velocity and angle), conditional_sparse_penalty | 结合目标距离门控：仅在距离 < threshold 时激活减速和姿态惩罚，否则忽略以允许自由调整。 |
| energy_efficiency | action (0-3) | 无能耗测量 | count_usage_penalty, action_magnitude (binary) | 可对动作非零施加小常数惩罚，或对主发动机 (action=2) 另加惩罚。需低权重或后期启用。 |
| terminal_settlement_bonus | 需要 done 和 info.success | done flag, info | sparse_terminal_reward | **不可实现**，弃用。 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高速撞击平台导致 crash | episode 结束前 y_velocity 负值大，x,y 接近零，未稳定双脚接触 | 增加 soft_landing 阶段的减速惩罚强度，或在接近目标时对主引擎使用加负权重；引入垂直接近速度限制。 |
| 永远不尝试点火，自由落体失败 | action 长期为 0，y 和 x 较少变化 | 调整能量效率惩罚过重，暂时移除或降低，确保到达奖励足够引导。 |
| 过度使用主引擎导致飞出视口 | 水平位置超界，大量 action=2 | 增加越界惩罚（可通过位置信息），或限制 main_engine 使用频度（若引擎不适用横向控制）。 |
| 姿态振荡，无法稳定接触 | body_angle 和 angular_velocity 高频摆动，双脚接触短暂 | 强化角度稳定惩罚（绝对值），加入角速度阻尼，必要时代码中进行姿态稳定专项训练。 |
| 仅单脚接触就停止，未形成稳定双脚 | 终止时 left/right 接触不同时为 1 | 在 soft_landing 职责中增加双脚同时接触的条件奖励，并惩罚单脚悬空状态。 |
| 悬停在目标附近但永不接触 | y 接近 0 但 y_velocity 未归于零，双脚未接触 | 引入接近零速度的奖励，并强化 soft_landing 中接触信号作用，或施加时间压力（后期可选）。 |

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + angular_velocity_penalty + contact_bonus + proximity_reward + velocity_penalty | 231.37 | 231.37 | 0.00 | 589.35 | angle_penalty=-0.003 angular_velocity_penalty=-0.001 contact_bonus=0.597 proximity_reward=1.723 velocity_penalty=-0.006 | target_solved_new_best |

```
