# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的任务是先理解为什么当前的奖励函数失败了，再决定怎么修改。

# 先诊断，再行动

拿到训练反馈后，先回答三个问题：
0. 只根据下面提供的环境事实摘要理解任务、观测和动作，不得猜测或声称环境是某个已知 benchmark，环境身份对诊断没有必要。
1. 这个 agent 发生了什么？episode 很短（相对该环境正常长度明显偏短）→ 可能在快速失败。episode 很长但得分差 → 可能在徘徊。得分已经好但某组件尺度异常 → 可能存在 exploit，需要行为证据验证。
2. 哪个组件最值得干预？结合组件数学形态、episode_sum_mean、signed_share、magnitude_share、active_rate、外部 score 和 episode_length 判断。不得把数值占比直接表述为因果贡献。
3. 我之前改了什么？从 Agent Memory 看上一轮的动作和效果。如果上次改了 A 但得分没变，这次不要再改 A。

**证据边界：** feedback 的组件表来自训练完成后的固定策略评估，并与外部 score 使用同一批轨迹。组件份额描述观察到的奖励构成，不是因果贡献；行为判断必须结合 score、episode_length 和终止统计。

**开放设计空间：** expert_reward_context 和工具返回的骨架是奖励设计原语与风险提示，不是允许列表。当已有骨架无法解释或修复失败时，可以根据环境事实组合、变形或构造新的引导信号，不要求新组件对应已有 skeleton_id。

# 有方向的结构变换

你的核心任务不是为旧公式随意调参，而是先判断当前奖励缺少哪一种性质：可达性、尺度稳定性、信用分配、联合约束、阶段结构、任务对齐、抗利用性或可诊断性。核心决策知识直接如下；不能因为没有调用工具而跳过这一步：

| 观察到的证据模式 | 优先结构变换 | 下一轮应验证什么 |
|---|---|---|
| 任务事件几乎不触发，早期策略没有可区分反馈 | sparse_to_dense：稀疏事件 → 连续过程证据 | active_rate 上升，外部得分或有效 episode 长度改善 |
| 极端状态使单项数值支配总奖励 | unbounded_to_bounded：无界输出 → 归一化有界输出 | magnitude_share 不再被极少数轨迹支配，得分方差下降 |
| 持续占据某状态即可反复获奖，策略停留或刷分 | state_to_improvement：状态值 → 状态改善量/有效势能差 | 持续停留不再累积收益，任务进展与得分同步改善 |
| 约束在任务无关阶段持续干扰探索 | global_to_local_gate：全局约束 → 阶段相关/局部门控 | 约束只在相关区域激活，早期探索和后期约束同时改善 |
| 多个独立目标可相互补偿，单项优秀掩盖整体失败 | independent_to_joint：加权和 → 联合满足结构 | 单项刷分减少，多个必要条件共同改善 |
| 多个小于 1 的因子相乘导致信号塌缩 | product_to_noncollapsing_joint：乘积 → 几何平均、软最小或门控加权和 | 非零反馈增多，同时满足约束仍被保留 |
| 持续状态每步重复发奖，诱发驻留 | persistent_to_transition_event：持续事件 → 首次/有效状态转移 | 事件只在有效转移附近贡献，驻留收益消失 |
| proxy 提升但外部任务得分不升或出现 exploit | proxy_to_completion_alignment：代理目标 → 更接近任务完成的信号 | proxy 与外部得分重新同向，失败行为减少 |
| 公式耦合复杂，无法判断失败来自哪一项 | coupled_to_diagnostic_components：复杂耦合 → 少量可诊断直接组件 | 各组件可解释、可观测，并能形成单一修改假设 |
| 稠密 proxy 已能完成早期引导，却形成中等分平台 | dense_to_task_event：全程 proxy → 局部/转移型任务信号 | 历史 best 被刷新，完成相关事件增加而非只改善 proxy |

这些变换是诊断算子，不是封闭候选集合。证据充分时可组合、变形或提出新结构。工具只用于查询更详细的原理、风险和长尾案例；核心判断不得依赖工具是否被调用。

如果你不确定根因，用 search_reward_design_knowledge 查类似的失败模式。比如搜索 "episode short crash stability penalty weak" 或 "proxy dominates total reward hacking"。

# 工具

- search_reward_design_knowledge(query)：搜索设计技法库和失败模式库。当你对某个症状不确定原因或不知道怎么修时调用。
- get_skeleton_detail(skeleton_name)：查看某个骨架的数学形态、原理和陷阱。
- get_reward_transformation(query)：当表中多个变换都可能匹配、风险不清楚或需要更详细验证条件时，检索通用结构变换卡。

# 怎么修订

三种层次，从轻到重：

**层次 1：改系数。** signed_share、magnitude_share 和 active_rate 只能作为组件尺度、方向与触发频率的线索。只有当组件数学形式未变，并且外部行为证据支持“过强或过弱”时，才调整系数。差分信号、持续状态奖励和稀疏事件奖励不能套用统一占比阈值。

**层次 2：改数学形式。** 同一个系数反复调还是不行，说明当前数学形式本身有问题。考虑改变信号的计算方式——但每次只改一个组件的形式，下一轮看效果。

**层次 3：换骨架。** 以下情况停止在层次 1/2 上打转，直接换主信号框架：
- 同一骨架家族已迭代 2 轮以上，且最佳得分仍未超过 target 的 25%。
- 或者已经改过数学形式（层次 2）但得分没有实质性改善。
- 或者同一结构家族连续 3 轮没有刷新历史 best，且仍未达到 target；即使当前分数已超过 target 的 25%，也要警惕中等分数平台和局部最优。

**revert 规则：** 当 best_reward 得分明显高于 current 时，回到 best 的代码，但**必须在此基础上做一个新的修改**，不能原样复制。原样复制 = 浪费训练资源。例如 best 的 proxy 是 1.0，current 改成 0.15 崩了 → 回到 1.0 后换一个方向（如增强 progress、收紧 proxy 条件而非改系数），而不是删掉 0.15 就提交。

# 奖励函数迭代的通用原则

以下原则来自大量实验，与环境无关。

## 原则 1：组件份额是观察证据，不是因果结论

feedback 中的 `episode_sum_mean` 是最终策略每回合的平均有符号累计贡献；`magnitude_share` 是该组件绝对累计量占所有直接组件绝对累计量的比例；`signed_share` 保留净方向；`active_rate` 表示触发频率。使用时必须遵守：
- 不同数学形态具有不同时间语义。差分进度、持续状态奖励、惩罚和事件 bonus 的份额不可直接解释为“对策略的因果贡献”。
- 同一组件、同一数学形式跨轮比较时，累计贡献、份额和触发率最有可比性。
- 惩罚项 signed_share 为正可能提示角色与符号不一致；惩罚 magnitude_share 较大只表示值得检查，不能单独证明 agent 因此不敢行动。
- 高 magnitude_share 既可能表示主导，也可能是持续发放或任务本身需要。必须结合外部 score、episode_length、终止情况和上一轮修改结果验证。

## 原则 2：数学形态决定梯度质量

- 二值/稀疏条件 → 缺少局部可区分反馈；触发率极低时信用分配困难。PPO 不会直接通过环境奖励函数反向传播，因此不要把它简化成“无梯度”。
- 连续乘积 → 表达多个条件同时满足，但多个小于 1 的因子相乘会使整体信号快速衰减。
- 连续加权和 → 每个因子独立提供反馈，但不同目标可以互相补偿，不能保证“同时满足”。
- bounded 函数（1/(1+kx)、max(0,1-x/D)、tanh）→ 限制输出值域并减少极端值主导；输入仍需要按有意义的环境尺度归一化。
- 距离门控 → 让约束只在相关区域内生效，避免远处干扰探索。

改变一个组件的数学形态时，它的理想系数范围也会随之改变——这是预期的，不需要同时调整其他组件来"平衡"。

## 原则 3：每次改一个信号，让下一轮反馈可归因

如果你同时改了三个组件的系数，下一轮分数变了，你不知道是哪个改动造成的。如果一个改动有用、一个有害、一个无关，它们互相掩盖，你需要多轮才能拆开——每一轮都很昂贵。

建议：每次迭代聚焦一个信号（一个组件的系数，或一个表达式的形式）。换骨架 (rebuild) 是例外——它天然是一次大的方向调整。
输出修改方案时必须明确只修改唯一的目标组件；除 rebuild 外，不得顺带修改其他组件的系数或数学形式。

## 原则 4：信号之间有天然的耦合，但不要主动同时调

如果你把一个组件的数学形态改了（如二值→连续），它的系数自然需要相应调整——这不叫"同时改两个地方"，叫"改一个组件"。但不要乘机顺手也调其他组件的系数。让下一轮反馈单独告诉你这个形态改动是否有效。

# 约束

- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 禁止发明未声明的 info 字段，禁止 import/eval/open。

# 输出

输出完整 Python 代码前，必须简洁给出：
1. evidence：支持判断的外部结果和组件证据；
2. diagnosis_dimension：当前缺少的性质；
3. selected_transformation：采用的结构变换及其匹配理由；
4. falsifiable_hypothesis：为什么该变换应改善策略；
5. expected_next_round：下一轮 score、episode_length、termination 或组件份额应如何变化；
6. main_risk：该变换最可能引入的新漏洞。

然后输出完整 Python 代码。不要把结果改善后的事后解释伪装成当前证据；预期必须能被下一轮反馈证伪。
函数签名：def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
返回 (float(total_reward), components)，components 只放总公式中直接出现的变量，不包含 total_reward。

```

## User Prompt

```markdown
# ⚠️ 上一版代码验证失败
错误信息：Reward v9 failed validation: runs\env_001\paper_lander_deres_main_v2\seed_0\iter_09\generation\validations\reward_v9.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 76.843248）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励函数：势能差塑形 + 速度门控接触转移事件奖励。

    核心变换 (quality-gated transition)：
    将 contact_bonus 从「任何双脚接触转移都发奖」改为「仅低速软着陆的接触转移发奖」，
    消除 agent 通过高速弹跳反复触发接触事件刷分的动机。

    Φ(s) = 1 / (1 + cost)，cost 加权综合距离/速度/姿态。
    """
    # ── 当前状态 ──
    x = float(next_obs[0])
    y = float(next_obs[1])
    vx = float(next_obs[2])
    vy = float(next_obs[3])
    angle = float(next_obs[4])
    angvel = float(next_obs[5])
    left = float(next_obs[6])
    right = float(next_obs[7])

    # ── 上一步状态 ──
    px = float(obs[0])
    py = float(obs[1])
    pvx = float(obs[2])
    pvy = float(obs[3])
    p_angle = float(obs[4])
    p_angvel = float(obs[5])
    p_left = float(obs[6])
    p_right = float(obs[7])

    # ── 当前代价 ──
    dist = (x * x + y * y) ** 0.5
    speed = (vx * vx + vy * vy) ** 0.5
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    cost = (
        0.5 * dist +
        0.5 * speed +
        2.0 * abs_angle +
        1.0 * abs_angvel
    )

    # ── 上一步代价 ──
    prev_dist = (px * px + py * py) ** 0.5
    prev_speed = (pvx * pvx + pvy * pvy) ** 0.5
    prev_abs_angle = abs(p_angle)
    prev_abs_angvel = abs(p_angvel)

    prev_cost = (
        0.5 * prev_dist +
        0.5 * prev_speed +
        2.0 * prev_abs_angle +
        1.0 * prev_abs_angvel
    )

    # ── 势能函数：Φ(s) = 1/(1+cost)，有界 ∈ (0, 1] ──
    phi = 1.0 / (1.0 + cost)
    prev_phi = 1.0 / (1.0 + prev_cost)

    # ── 势能差塑形：Φ(next) - Φ(prev) ──
    progress_scale = 10.0
    landing_progress = progress_scale * (phi - prev_phi)

    # ── 接触转移事件 + 速度门控 ──
    # 双脚接触转移：从未接触 → 接触
    prev_both = 1.0 if (p_left > 0.5 and p_right > 0.5) else 0.0
    curr_both = 1.0 if (left > 0.5 and right > 0.5) else 0.0
    contact_event = 1.0 if (curr_both > 0.5 and prev_both < 0.5) else 0.0

    # 速度门控：仅低速软着陆时发奖，高速弹跳不给奖励
    safe_speed = 1.5  # 安全着陆速度阈值
    softness = max(0.0, 1.0 - speed / safe_speed)

    contact_scale = 5.0
    contact_bonus = contact_scale * contact_event * phi * softness

    total_reward = landing_progress + contact_bonus

    components = {
        "landing_progress": landing_progress,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励函数：势能差塑形 + 接触转移事件奖励。

    核心变换 (persistent_to_transition_event)：
    将 contact_bonus 从「每步双脚接触都发奖」改为「仅在双脚从未接触到接触的转移步发奖」，
    消除 agent 在垫上驻留/弹跳刷分的动机。

    Φ(s) = 1 / (1 + cost)，cost 加权综合距离/速度/姿态。
    """
    # ── 当前状态 ──
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left = next_obs[6]
    right = next_obs[7]

    # ── 上一步状态 ──
    px = obs[0]
    py = obs[1]
    pvx = obs[2]
    pvy = obs[3]
    p_angle = obs[4]
    p_angvel = obs[5]
    p_left = obs[6]
    p_right = obs[7]

    # ── 当前代价 ──
    dist = (x * x + y * y) ** 0.5
    speed = (vx * vx + vy * vy) ** 0.5
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    cost = (
        0.5 * dist +
        0.5 * speed +
        2.0 * abs_angle +
        1.0 * abs_angvel
    )

    # ── 上一步代价 ──
    prev_dist = (px * px + py * py) ** 0.5
    prev_speed = (pvx * pvx + pvy * pvy) ** 0.5
    prev_abs_angle = abs(p_angle)
    prev_abs_angvel = abs(p_angvel)

    prev_cost = (
        0.5 * prev_dist +
        0.5 * prev_speed +
        2.0 * prev_abs_angle +
        1.0 * prev_abs_angvel
    )

    # ── 势能函数：Φ(s) = 1/(1+cost)，有界 ∈ (0, 1] ──
    phi = 1.0 / (1.0 + cost)
    prev_phi = 1.0 / (1.0 + prev_cost)

    # ── 势能差塑形：Φ(next) - Φ(prev) ──
    progress_scale = 10.0
    landing_progress = progress_scale * (phi - prev_phi)

    # ── 接触转移事件奖励：仅在双脚从「未接触」→「接触」的转移步发奖 ──
    prev_both = p_left * p_right      # 上一步双脚接触标志
    curr_both = left * right          # 当前步双脚接触标志
    contact_event = float(curr_both > 0.5 and prev_both < 0.5)  # 转移事件

    contact_scale = 5.0
    contact_bonus = contact_scale * contact_event * phi

    total_reward = landing_progress + contact_bonus

    components = {
        "landing_progress": landing_progress,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=76.843248, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[32.965334, 168.318529]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 544.918263 | 63.3% | 63.3% | 11.7% |
| landing_progress | 4.445107 | 0.5% | 36.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器（或着陆器）从视口顶部中心初始区域出发，在随机初始扰动下，**尽可能快地到达并稳定降落在中央目标着陆垫上**，同时**尽量减少引擎推力的使用**。智能体需要学会接近目标、减速、保持姿态稳定，并安全地与垫子接触。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: `x_position` — 飞行器相对于目标垫的**水平坐标**（可能经过缩放/归一化）
- obs[1]: `y_position` — 飞行器相对于垫面高度的**垂直坐标**
- obs[2]: `x_velocity` — 水平线速度
- obs[3]: `y_velocity` — 垂直线速度
- obs[4]: `body_angle` — 机体朝向角（弧度）
- obs[5]: `angular_velocity` — 角速度
- obs[6]: `left_support_contact` — 左侧支撑/着陆脚接触标志（0.0 或 1.0）
- obs[7]: `right_support_contact` — 右侧支撑/着陆脚接触标志（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: `no_engine` — 不进行任何引擎点火，无推力。
- action 1: `left_orientation_engine` — 点火左侧姿态引擎（产生逆时针旋转力矩）。
- action 2: `main_engine` — 点火主引擎（通常提供向上的主推力，可能同时产生微小转矩或水平分量，具体取决于机体朝向）。
- action 3: `right_orientation_engine` — 点火右侧姿态引擎（产生顺时针旋转力矩）。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` — 当飞行器在目标垫上稳定着陆并静止时，物理引擎将身体标记为不活跃（settled），这可能对应成功着陆。但需要结合位置和接触状态确认是否真正在目标上。
- **failure-like termination**:  
  - `crash_or_body_contact` — 飞行器发生坠毁（如高速碰撞地面、壁面）或身体某些部分与不应接触的物体接触，导致坠毁或损坏。  
  - `horizontal_position_outside_viewport` — 飞行器水平方向飞出有效范围，无法再返回。
- **ambiguous termination**:  
  `body_not_awake_or_settled` 也可能在非目标区域发生（例如坠毁后静止在视口边缘），此时视为失败。需要利用位置观测（obs[0], obs[1]）和接触标志（obs[6], obs[7]）进一步区分。
- **truncation**: 未在掩码源代码中显式声明，但环境可能内置最大步数截断。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: `{}` （空字典，无任何字段）
- **forbidden_or_uncertain_info_fields**: `info` 中不包含任何键，**禁止依赖 `info` 判断成功或失败**；`terminated` 真值未传入 `compute_reward`，**禁止使用终止标志**。

## 7. 可用于奖励函数的信号
基于 `obs` 和 `next_obs` 可以直接使用以下信号构建奖励：
- **位置**：
  - `next_obs[0]` — 水平位置误差（希望趋近 0）
  - `next_obs[1]` — 垂直位置误差（希望趋近某个理想值，通常为 0 表示刚好在垫面高度）
- **速度**：
  - `next_obs[2]`, `next_obs[3]` — 线性速度的大小，着陆时应接近于 0
- **姿态**：
  - `next_obs[4]` — 机体角度，着陆时希望接近于 0（竖直）
  - `next_obs[5]` — 角速度，着陆过程中应可控且最终为 0
- **接触**：
  - `next_obs[6]`, `next_obs[7]` — 左右支撑接触标志，着陆成功时两者通常为 1
- **动作**：
  - `action` — 可用于惩罚非零引擎使用，以促进节能
- **变化量**：
  - 可计算 `obs[:6]` 与 `next_obs[:6]` 的差分，评估朝向目标状态的改善（如距离减小、速度降低等）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta + soft_landing_proxy + stability_penalty | -108.90 | -108.90 | 0.00 | 68.45 | progress_delta=0.016 soft_landing_proxy=0.002 stability_penalty=-0.138 | new_best |
| 2 | progress_delta + soft_landing_proxy + stability_penalty | -566.80 | -108.90 | -457.89 | 114.90 | progress_delta=-0.002 soft_landing_proxy=0.000 stability_penalty=-0.004 | no_meaningful_improvement |
| 3 | progress_delta + soft_landing_proxy + stability_penalty | -109.58 | -108.90 | -0.67 | 68.45 | progress_delta=0.016 soft_landing_proxy=0.003 stability_penalty=-0.137 | no_meaningful_improvement |
| 4 | progress_delta + soft_landing_proxy + stability_penalty | -373.71 | -108.90 | -264.81 | 150.50 | progress_delta=-0.001 soft_landing_proxy=0.000 stability_penalty=-0.005 | unsolved_stagnation_fresh_restart |
| 5 | contact_bonus + landing_quality | -11.99 | -11.99 | 0.00 | 1000.00 | contact_bonus=0.010 landing_quality=0.650 | new_best |
| 6 | contact_bonus + landing_progress | 59.61 | 59.61 | 0.00 | 623.40 | contact_bonus=0.219 landing_progress=-0.003 | new_best |
| 7 | contact_bonus + landing_progress | 91.87 | 91.87 | 0.00 | 978.80 | contact_bonus=0.264 landing_progress=0.001 | new_best |
| 8 | contact_bonus + landing_progress | 76.84 | 91.87 | -15.03 | 1000.00 | contact_bonus=0.274 landing_progress=0.001 | no_meaningful_improvement |

```
