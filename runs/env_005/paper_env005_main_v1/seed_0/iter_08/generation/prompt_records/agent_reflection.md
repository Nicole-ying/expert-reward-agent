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
# ⚠️ 上一版代码验证失败
错误信息：Reward v1 failed validation: 出现未允许的 obs/next_obs 切片 (record: runs/env_005/paper_env005_main_v1/seed_0/iter_08/generation/validations/reward_v1.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Four-legged locomotion reward v1.
    Core design hypothesis: focus on height management and foot contact rhythm.
    Forward velocity is deliberately absent from v1 to prevent fast-then-fail collapse.
    """

    # ---------- Extract next observation signals ----------
    # Body height
    body_z = next_obs[0]

    # Body orientation (quaternion)
    quat_w = next_obs[4]  # scalar component
    quat_x = next_obs[1]
    quat_y = next_obs[2]
    quat_z = next_obs[3]

    # Body velocities
    forward_vel = next_obs[13]   # desired forward movement
    lateral_vel = next_obs[14]   # sideways drift
    vertical_vel = next_obs[15]  # bouncing

    # Angular velocities (body rotation)
    roll_rate = next_obs[16]
    pitch_rate = next_obs[17]

    # Foot contact forces (z-component of each foot's contact)
    foot_contacts = next_obs[45:51]    # front-left foot
    foot_contacts_2 = next_obs[63:69]  # front-right foot
    foot_contacts_3 = next_obs[81:87]  # rear-left foot
    foot_contacts_4 = next_obs[99:105] # rear-right foot

    # Foot z-contact force (index 2 in the 6D vector)
    fl_foot_z = foot_contacts[2]
    fr_foot_z = foot_contacts_2[2]
    rl_foot_z = foot_contacts_3[2]
    rr_foot_z = foot_contacts_4[2]

    # ---------- Component 1: Height Management (主学习信号) ----------
    # Role: core driving force — stay standing
    # Mathematical form: symmetric penalty around target height with deadzone
    target_height = 0.6
    height_error = body_z - target_height

    # Symmetric penalty: penalizes both being too low (collapsing) and too high (jumping)
    # Deadzone of +/-0.05 to allow natural body oscillation
    height_deviation = abs(height_error) - 0.05
    if height_deviation < 0.0:
        height_deviation = 0.0

    # Linear penalty beyond deadzone, capped to avoid explosion
    height_penalty = 4.0 * min(height_deviation, 0.4)
    height_management = -height_penalty

    # ---------- Component 2: Foot Contact Regularization (步态引导) ----------
    # Role: gait regularization — encourage rhythmic foot contact patterns
    # Use tanh on contact forces to convert to ~[0,1] contact indicator
    # Then reward having at least two feet on ground (diagonal pairs preferred)

    contact_threshold = 50.0  # minimum contact force to consider "grounded"
    sensitivity = 0.05

    def soft_contact(force, threshold, sens):
        """Smooth indicator: 0 when force << threshold, 1 when force >> threshold."""
        return 0.5 + 0.5 * (2.718281828 ** ((force - threshold) * sens) - 2.718281828 ** (-(force - threshold) * sens)) / (2.718281828 ** ((force - threshold) * sens) + 2.718281828 ** (-(force - threshold) * sens))

    # If we can't compute tanh easily, use a simpler sigmoid-like function
    fl_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (fl_foot_z - contact_threshold))))
    fr_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (fr_foot_z - contact_threshold))))
    rl_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (rl_foot_z - contact_threshold))))
    rr_contact = 1.0 / (1.0 + (2.718281828 ** (-sensitivity * (rr_foot_z - contact_threshold))))

    # Total ground contact (0 = no feet, 4 = all feet on ground)
    total_contact = fl_contact + fr_contact + rl_contact + rr_contact

    # Reward: prefer 2-3 feet on ground (walking), penalize 0-1 (falling) or 4 (standing still)
    # Use Gaussian-like reward centered at 2.5 (dynamic gait)
    contact_target = 2.5
    contact_deviation = abs(total_contact - contact_target)
    contact_reward = 1.5 * (2.718281828 ** (-2.0 * (contact_deviation ** 2)))

    # ---------- Component 3: Body Stability (姿态约束) ----------
    # Role: stability constraint — keep body upright and suppress roll/pitch
    # Compute body "up" vector from quaternion (projection of body z-axis onto world z-axis)
    # For a quaternion (x, y, z, w), the body z-axis in world frame is:
    # world_z_body = 1 - 2*(x^2 + y^2)  (when quaternion represents body-to-world rotation)
    body_up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # Reward closeness to 1.0 (perfectly upright)
    # Penalize deviation from upright posture
    tilt_penalty = 1.5 * (1.0 - body_up_z)

    # Also penalize angular velocity (roll + pitch rate)
    angular_rate_penalty = 0.3 * (abs(roll_rate) + abs(pitch_rate))

    stability_cost = tilt_penalty + angular_rate_penalty

    # ---------- Component 4: Forward Motivation (弱前进信号) ----------
    # Role: weak forward bias — gentle encouragement to move forward
    # Very small weight to avoid fast-then-fail; height+contact are primary
    forward_encouragement = 0.3 * forward_vel

    # ---------- Component 5: Lateral Drift Penalty (侧向抑制) ----------
    # Role: suppress side-to-side sway
    lateral_penalty = 0.6 * abs(lateral_vel)

    # ---------- Assemble total reward ----------
    total_reward = (
        height_management
        + contact_reward
        - stability_cost
        + forward_encouragement
        - lateral_penalty
    )

    # ---------- Components dictionary ----------
    components = {
        'height_management': height_management,
        'contact_reward': contact_reward,
        'stability_cost': -stability_cost,
        'forward_encouragement': forward_encouragement,
        'lateral_penalty': -lateral_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **height_management（高度管理）- 主学习信号**
   - 角色：核心驱动力，告诉agent“保持站立”
   - 数学形态：对称惩罚 `-w * soft_clip(|body_z - target| - deadzone)`
   - 为什么选它为主信号：历史实验显示所有前向速度主导的设计都因高度违规而崩溃（19/20失败），所以v1必须让高度管理成为最强信号

2. **contact_reward（足部接触正则化）- 步态引导**
   - 角色：鼓励形成有节奏的步态（2-3脚着地），既不死站也不空踩
   - 数学形态：sigmoid软接触检测 → 高斯形奖励（中心2.5，惩罚0/1或4脚着地）
   - 这是全新信号，之前设计从未使用过脚部接触信息

3. **stability_cost（身体稳定性）- 姿态约束**
   - 角色：保持躯体直立，抑制翻滚/俯仰
   - 包含两个子项：`tilt_penalty`（基于四元数计算身体竖直投影）和`angular_rate_penalty`（角速度惩罚）

4. **forward_encouragement（弱前进信号）- 前向偏置**
   - 角色：极轻的前进鼓励，权重仅0.3
   - 避免历史设计中forward_velocity占42.5% signed_share导致“不顾姿态向前冲”的问题

5. **lateral_penalty（侧向抑制）- 抑制侧摆**
   - 角色：惩罚左右晃动

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- explicit_success_flag_available = false，无显式成功终止标志
- explicit_failure_flag_available = false，无显式失败终止标志
- 所有终止信号都在高度范围检查中隐式产生，v1通过height_management连续引导替代离散惩罚

## 留到后续迭代的组件

- **energy_penalty**：v1阶段agent应先学会站立和行走，动作效率优化后续加入
- **action_smoothness_penalty**：需要动作历史，目前信号偏向躯体控制，动作平滑后加
- **alive_bonus**：可能诱导hover_or_stand_still，需等agent能稳定前进后再考虑

## 训练后应观察的failure mode

1. **stand_still_collapse**：agent可能找到静态站立（4脚着地+接触奖励≈2.5附近）的局部最优，不前移
   - 缓解：forward_encouragement权重小但持续存在，应引导缓慢前移
   - 如果出现：v2增加前进信号权重或引入progress-based奖励

2. **oscillatory_gait**：接触正则化可能诱导高频抬腿放腿（快速切换2→3→2脚着地）而非稳定行走
   - 缓解：没有显式奖励高频动作，自然动力学应惩罚无效振动
   - 如果出现：v2加入步频约束或基于速度的gate

3. **height_overcorrection**：高度管理的对称惩罚可能导致agent跳跃（body_z>>target也是小惩罚）
   - 缓解：deadzone 0.05允许小幅波动，vertical_vel未单独惩罚但angular_rate会抑制弹跳
   - 如果出现：v2加入vertical_velocity惩罚或改为非对称惩罚（高于target惩罚更轻）

# Search objective
- target_score: 6000.000000
- current_score: 12.313339
- gap_to_target: 5987.686661
- target_achievement_ratio: 0.205%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 12.313339）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Indices for next_obs
    BODY_X_VEL = 13
    BODY_Y_VEL = 14
    ROLL_VEL = 16
    PITCH_VEL = 17

    # Quaternion (w, x, y, z)
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    body_z = next_obs[0]
    forward_vel = next_obs[BODY_X_VEL]
    lateral_vel = next_obs[BODY_Y_VEL]
    roll_vel = next_obs[ROLL_VEL]
    pitch_vel = next_obs[PITCH_VEL]

    # Weights
    w_fwd = 1.0
    w_orient = 0.3
    w_lat = 0.2
    w_ang = 0.05
    w_height = 2.0

    # Boundary-hinge height penalty: safe zone [0.3, 0.9]
    height_too_low = max(0.0, 0.3 - body_z)
    height_too_high = max(0.0, body_z - 0.9)
    height_boundary_penalty = -w_height * (height_too_low + height_too_high)

    # Reward components
    forward_reward = w_fwd * forward_vel
    orientation_reward = w_orient * body_up_z
    lateral_penalty = -w_lat * abs(lateral_vel)
    angular_penalty = -w_ang * (abs(roll_vel) + abs(pitch_vel))

    total_reward = (
        forward_reward
        + orientation_reward
        + lateral_penalty
        + angular_penalty
        + height_boundary_penalty
    )

    components = {
        "forward_velocity_reward": forward_reward,
        "orientation_reward": orientation_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty,
        "height_boundary_penalty": height_boundary_penalty,
    }
    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Materially different reward design (v2) — adds height deviation penalty
    to prevent body collapse, which caused 19/20 early terminations in v1.
    All v1 components and weights are preserved unchanged.
    """
    # next_obs indices
    BODY_X_VEL = 13
    BODY_Y_VEL = 14
    ROLL_VEL = 16
    PITCH_VEL = 17

    # Quaternion components: assumed order (w, x, y, z)
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # Body height (vertical position)
    body_z = next_obs[0]

    forward_vel = next_obs[BODY_X_VEL]
    lateral_vel = next_obs[BODY_Y_VEL]
    roll_vel = next_obs[ROLL_VEL]
    pitch_vel = next_obs[PITCH_VEL]

    # Weights — v1 weights preserved
    w_fwd    = 1.0
    w_orient = 0.3
    w_lat    = 0.2
    w_ang    = 0.05

    # New height deviation penalty
    # Healthy range: [0.2, 1.0], center = 0.6, half-range = 0.4
    # At body_z=0.2 or 1.0, penalty = -1.0; at center 0.6, penalty = 0
    w_height = 1.0
    height_deviation = (body_z - 0.6) / 0.4
    height_deviation_penalty = -w_height * (height_deviation ** 2)

    # Components (v1 preserved)
    forward_reward = w_fwd * forward_vel
    orientation_reward = w_orient * body_up_z
    lateral_penalty = -w_lat * abs(lateral_vel)
    angular_penalty = -w_ang * (abs(roll_vel) + abs(pitch_vel))

    total_reward = (
        forward_reward
        + orientation_reward
        + lateral_penalty
        + angular_penalty
        + height_deviation_penalty
    )

    components = {
        "forward_velocity_reward": forward_reward,
        "orientation_reward": orientation_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty,
        "height_deviation_penalty": height_deviation_penalty,
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=12.313339, len=161.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-181.890465, 175.626337]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 158.515021 | 53.4% | 63.7% | 100.0% |
| orientation_reward | 46.309150 | 15.6% | 15.6% | 100.0% |
| lateral_velocity_penalty | -23.634231 | -8.0% | 8.0% | 100.0% |
| angular_velocity_penalty | -20.519881 | -6.9% | 6.9% | 100.0% |
| height_deviation_penalty | -17.426705 | -5.9% | 5.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 3D 四足机器人，通过 8 个扭矩控制关节，实现持续、稳定的前向行走或奔跑，同时保持身体直立。机器人身体高度必须保持在健康范围内（0.2~1.0），过低（倒塌）或过高（弹飞）都会导致回合终止。核心目标是维持稳定的前向运动，而不是仅仅保持站立。

## 3. 观察空间 observation_space
- type: Box
- shape: [105]
- dtype: float
- obs[0]: body_z – 主体垂直位置
- obs[1:5]: body_orientation（四元数，共 4 维）– 主体姿态
- obs[5:13]: joint_angles（共 8 维）– 8 个关节角度
- obs[13]: body_x_velocity – 主体世界 x 轴速度（前向速度）
- obs[14]: body_y_velocity – 主体世界 y 轴速度（横向速度）
- obs[15]: body_z_velocity – 主体垂直速度
- obs[16]: body_roll_velocity – 滚转角速度
- obs[17]: body_pitch_velocity – 俯仰角速度
- obs[18]: body_yaw_velocity – 偏航角速度
- obs[19:27]: joint_velocities（共 8 维）– 8 个关节角速度
- obs[27:33]: torso 接触力/力矩（fx, fy, fz, tx, ty, tz）
- obs[33:39]: front_left upper leg 接触力/力矩
- obs[39:45]: front_left lower leg 接触力/力矩
- obs[45:51]: front_left foot 接触力/力矩
- obs[51:57]: front_right upper leg 接触力/力矩
- obs[57:63]: front_right lower leg 接触力/力矩
- obs[63:69]: front_right foot 接触力/力矩
- obs[69:75]: back_left upper leg 接触力/力矩
- obs[75:81]: back_left lower leg 接触力/力矩
- obs[81:87]: back_left foot 接触力/力矩
- obs[87:93]: back_right upper leg 接触力/力矩
- obs[93:99]: back_right lower leg 接触力/力矩
- obs[99:105]: back_right foot 接触力/力矩

## 4. 动作空间 action_space
- type: Box (continuous)
- shape: [8]
- 范围: [-1.0, 1.0]
- action 0: hip_1_torque – 第一个髋关节扭矩
- action 1: ankle_1_torque – 第一个踝关节扭矩
- action 2: hip_2_torque – 第二个髋关节扭矩
- action 3: ankle_2_torque – 第二个踝关节扭矩
- action 4: hip_3_torque – 第三个髋关节扭矩
- action 5: ankle_3_torque – 第三个踝关节扭矩
- action 6: hip_4_torque – 第四个髋关节扭矩
- action 7: ankle_4_torque – 第四个踝关节扭矩

（注：四足共 8 个驱动关节，对称分布；每腿由髋 + 踝关节组成）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式的成功终止条件；任务要求持续前进，只有时间截断（truncation）可能对应完成整个 episode
- failure-like termination:
  - body_height_outside_healthy_range（主体 z 超出 0.2~1.0）：倒塌或弹飞
  - state_value_outside_finite_range（任何状态量为 NaN 或 inf）
- ambiguous termination: 无
- truncation: time_limit_reached（达到最大步数限制）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []  （环境 step 返回的 info 为空 dict，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin, x_velocity, y_velocity 等均被禁止

## 7. 可用于奖励函数的信号
- position:
  - body_z（obs[0]）– 垂直高度，可用于惩罚离健康范围偏差
- velocity:
  - body_x_velocity（obs[13]）– 前向速度，可直接作为前进奖励
  - body_y_velocity（obs[14]）– 横向速度，可用于抑制侧向漂移
  - body_z_velocity（obs[15]）– 垂直速度，结合高度变化可判断坠落倾向
- orientation:
  - body_orientation 四元数（obs[1:5]）– 可用于保持身体水平，惩罚倾斜
  - roll/pitch/yaw 角速度（obs[16:18]）– 可惩罚快速旋转，维持稳定姿态
- contact:
  - 各腿足部接触力（obs[45:51], [63:69], [81:87], [99:105]）– 判断地面接触，可用于奖励足部触地或惩罚不稳定的接触模式
- action/engine:
  - 关节扭矩 action[0:7] – 可用于惩罚过大扭矩，提高能效
  - 关节速度（obs[19:27]）– 可结合扭矩计算功率

# Compact expert route context
- selected_route_id: locomotion_continuous_control
- recommended_design_roles: forward_progress_reward (前进/速度奖励), terminal_failure_penalty (失败惩罚), energy_penalty (动作能耗惩罚), alive_bonus (存活奖励), action_smoothness_penalty (动作平滑惩罚), stability_penalty (姿态/稳定性惩罚)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.
```
