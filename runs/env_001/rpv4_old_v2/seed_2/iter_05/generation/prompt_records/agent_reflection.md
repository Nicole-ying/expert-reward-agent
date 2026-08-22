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

User prompt 里的 #4 Subagent 诊断报告已经替你完成了三件事，从这里出发：

- **骨架演进表** — 并排了同一骨架各轮的 score、len、每个组件的 ep_sum 和 active_rate。纵向扫一眼就能看到："改了 A 那轮，B 塌了，score 跌了"。这是你的核心决策依据。
- **组件健康度** — 每个组件一行判定（健康/死组件/淹没/错位），告诉你哪个信号在驱动 agent、哪个形同虚设。
- **干预建议** — "最需要干预的组件是 X，因为 per-step 量级只有 Y 的 1/50"。

结合 per-step 量级（`ep_sum ÷ len`），回答两个问题：

1. **agent 学到了什么行为？**
2. **哪个组件最值得干预？** 不一定是变化最大的——如果 A 涨了 B 崩了，解释因果链。

一次只选择一个干预目标。current 明显差于 best 时，以 best 代码为基础做最小修改。

## 2. 信号完备性检查

检查当前奖励是否具有任务所需且可达的职责，而不是要求固定组件名称：

- 任务进展或可学习的过程引导；
- 必要的稳定、安全或动作约束；
- 当过程最优不等于任务完成时，能区分联合满足或完成状态的信号。

## 3. 选择干预层级

Subagent 诊断报告会明确告诉你哪个组件最需要干预。**它的推荐是你的起点。** 如果你不同意它的推荐，你必须在 response 里解释为什么不采纳。

先用 `episode_sum_mean ÷ len` 算出每个组件的 **per-step 量级**，对比谁在真正驱动行为。

然后逐组件诊断，每次只改一个组件：

**如果 per-step 量级明显失衡，但数学形态合理** → 调系数。强的组件在产生健康行为（score 涨、terminated 涨）就保留它、提弱组件；强的组件在驱动病态行为就削弱它。

**如果某个组件数学形态本身传达了错误激励** → 改这个组件的公式形式。典型信号：值在训练全程几乎不变（死组件）、乘积塌缩导致 active_rate 接近零、持续状态值无界累积导致 agent 赖着不走、全局惩罚全覆盖压制探索。参考变换方向：

| 证据 | 变换 |
|---|---|
| 事件几乎不触发 | 稀疏 → 连续 |
| 极端值支配 | 无界 → 有界 |
| 占据状态即获奖 | 状态值 → 改善量 |
| 乘积塌缩 | 乘积 → 加性/几何平均 |
| proxy 与外部分数脱钩 | 代理 → 任务完成对齐 |

**如果多轮迭代从未拿到可接受分数，且 Subagent 诊断和 Memory 表确认当前骨架方向错误** → 换主信号框架。选这一步前必须确认：历史上是否已有迭代拿到过不错的分数？如果有——你不是需要换框架，你需要在 best 基础上做上述两种修改之一。

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
错误信息：Reward v1 failed validation: 出现禁止模式: import ; 修复要求：删除所有import和numpy调用；平方根使用**0.5，有界函数使用max/min或不需要库的代数表达式 (record: runs/env_001/rpv4_old_v2/seed_2/iter_05/generation/validations/reward_v1.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# reward_v1.py

```python
import math

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lander navigation task.
    Roles: goal_distance_progress (improvement_delta of distance to pad),
           soft_landing (proximity‑gated velocity penalty),
           orientation_penalty (quadratic on body angle),
           landing_contact_success (soft completion proxy).
    """
    # ------------------------------------------------------------------
    # Unpack observations
    # ------------------------------------------------------------------
    prev_x, prev_y = obs[0], obs[1]
    prev_vx, prev_vy = obs[2], obs[3]
    # prev_angle = obs[4]          # not used directly
    # prev_angvel = obs[5]        # not used
    # prev_left  = obs[6], prev_right = obs[7]  # contact flags of current step are not needed

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    # next_angvel = next_obs[5]   # not used
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # ------------------------------------------------------------------
    # Distance to target pad  (0,0)
    # ------------------------------------------------------------------
    dist_prev = math.sqrt(prev_x**2 + prev_y**2)
    dist_next = math.sqrt(next_x**2 + next_y**2)

    # ------------------------------------------------------------------
    # 1. Goal distance progress  (improvement_delta)
    #    Positive when moving towards the pad, negative when retreating.
    # ------------------------------------------------------------------
    w_progress = 5.0
    progress = w_progress * (dist_prev - dist_next)

    # ------------------------------------------------------------------
    # 2. Soft landing  (proximity‑gated velocity penalty)
    #    Large penalty for speed when close to the pad, small when far.
    # ------------------------------------------------------------------
    w_vel = 1.0
    k_dist = 20.0   # controls how sharply the gate opens near the pad
    proximity_gate = 1.0 / (1.0 + k_dist * dist_next**2)
    vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate

    # ------------------------------------------------------------------
    # 3. Orientation penalty  (quadratic penalty on tilt)
    #    Body angle should be near zero (upright).
    # ------------------------------------------------------------------
    w_angle = 0.5
    angle_penalty = -w_angle * (next_angle**2)

    # ------------------------------------------------------------------
    # 4. Soft completion proxy  (both legs on pad + very low speed)
    #    Gives a positive boost when the lander appears to have settled.
    # ------------------------------------------------------------------
    w_contact = 2.0
    v_thresh = 0.1
    legs_on_ground = (next_left_contact >= 0.5) and (next_right_contact >= 0.5)
    speed_low = (abs(next_vx) < v_thresh) and (abs(next_vy) < v_thresh)
    contact_bonus = w_contact if (legs_on_ground and speed_low) else 0.0

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = progress + vel_penalty + angle_penalty + contact_bonus

    components = {
        "goal_distance_progress": progress,
        "soft_landing":          vel_penalty,
        "orientation_penalty":   angle_penalty,
        "landing_contact_success": contact_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **任务家族 / 动力学子类型**  
  `navigation_goal_reaching` / `goal_approach_and_soft_contact`（2D 着陆器，离散控制，双腿着陆）。

- **选中的奖励职责**  
  - 主学习信号：`goal_distance_progress`  
  - 稳定/安全约束：`soft_landing`、`orientation_penalty`  
  - 任务完成近似信号：`landing_contact_success`

- **职责 → 信号映射及公式算子**  
  | 职责 | 信号 | 算子 |
  |------|------|------|
  | `goal_distance_progress` | `x_position`, `y_position` → 当前与下一步距离 | `improvement_delta`（期望距离减小） |
  | `soft_landing` | `x_velocity`, `y_velocity` + 距离 | `proximity_gated` 速度惩罚（二次速度 × gate） |
  | `orientation_penalty` | `body_angle` | `quadratic_penalty`（‑w * angle²） |
  | `landing_contact_success` | `left_support_contact`, `right_support_contact`, `x_velocity`, `y_velocity` | `binary_AND_gate` + `velocity_threshold` → 固定正奖励 |

- **排除的职责及原因**  
  - `fuel_efficiency`（`action`）：v1 阶段优先保证成功着陆，过早惩罚引擎使用会压制探索，留待后续迭代。  
  - `angular_velocity_smoothing`：不是核心约束，v1 先关注姿态角本身。  
  - `time_efficiency_penalty`：无 episode 长度截断，固定步惩罚可能干扰着陆策略。  
  - `hard_success_flag`：环境未提供显式 `info` 成功标志，不可用。  
  - `explicit_episodic_time_limit`：同样不可用。

- **为什么没有 `terminal_success_reward` / `terminal_failure_penalty`**  
  环境卡片明确 `explicit_success_flag_available=false`、`explicit_failure_flag_available=false`，`info` 为空。不能也无依据构造一次性终端奖励。

- **留到后续迭代的职责**  
  - `fuel_efficiency`（加入极小动作代价，但需避免提前压制推力）。  
  - `angular_velocity_smoothing`（若训练中出现姿态振荡则加入）。  
  - `time_efficiency_penalty`（若观察到大量悬停延迟行为，可加入极小的每步惩罚）。  
  - 更复杂的 `curriculum_weighting` 或多阶段门控（如根据训练进度调节速度门限）。

- **训练后应观察的 failure modes**  
  - **高速撞击目标垫**：y_velocity 仍为较大负值便落地，可能被 `soft_landing` gate 捕获不足；需收紧 gate 的 `k_dist` 或增大速度惩罚权重。  
  - **单腿着陆侧翻**：`landing_contact_success` 仅奖励双腿，若 agent 只放一条腿则得不到 bonus；可增加对单腿接触的负向提示（未来迭代）。  
  - **停滞不动**：如果 `progress` 分量始终为零且动作均选 `no_engine`，说明缺乏探索激励；可加入极小动作熵奖励或增大 `w_progress`。  
  - **姿态摆动**：若 `orientation_penalty` 不够强，agent 可能反复倾斜再纠正；检查角度惩罚量级，必要时加入 `angular_velocity` 惩罚。  
  - **引擎浪费（无意义的左/右引擎）**：目前未惩罚，若 agent 频繁闪烁方向引擎但位置无变化，可在下一版加入小权重动作代价。

# 1. Search objective
- target_score: 200.000000
- current_score: -120.253372
- gap_to_target: 320.253372
- target_achievement_ratio: -60.127%

# 2. 上一轮奖励函数代码（该轮得分: -120.253372）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation (using next_obs, the state after action)
    x       = next_obs[0]   # horizontal offset from target pad center
    y       = next_obs[1]   # vertical offset from target pad
    vx      = next_obs[2]   # horizontal velocity
    vy      = next_obs[3]   # vertical velocity
    angle   = next_obs[4]   # body tilt angle (rad)
    angvel  = next_obs[5]   # angular velocity
    left_c  = next_obs[6]   # left leg contact
    right_c = next_obs[7]   # right leg contact

    # ------------------------------------------------------------------
    # Component A - goal proximity (positive, bounded approach reward)
    # ------------------------------------------------------------------
    w_prox = 2.0
    dist = (x**2 + y**2) ** 0.5
    # bounded reward: 1.0 when dist=0, decays to 0 when dist large
    comp_prox = w_prox * (1.0 - dist / (dist + 1.0))

    # ------------------------------------------------------------------
    # Component B - velocity penalty (soft brake)
    # ------------------------------------------------------------------
    w_vel = 0.5
    comp_vel = w_vel * (-vx**2 - vy**2)

    # ------------------------------------------------------------------
    # Component C - angle penalty
    # ------------------------------------------------------------------
    w_ang = 1.0
    comp_angle = w_ang * (-angle**2 - 0.1 * angvel**2)

    # ------------------------------------------------------------------
    # Component D - contact quality (leg contact gated by distance)
    # ------------------------------------------------------------------
    w_contact = 2.0
    gate = 1.0 / (1.0 + dist)   # near pad -> ~1, far away -> small
    leg_contact = left_c * right_c   # bilateral contact only
    comp_contact = w_contact * leg_contact * gate

    # ------------------------------------------------------------------
    # Component E - soft success shaping (sum of linear scores)
    # ------------------------------------------------------------------
    w_soft = 30.0

    tol_x     = 0.1
    tol_y     = 0.1
    tol_vx    = 0.2
    tol_vy    = 0.2
    tol_angle = 0.2

    score_x     = max(0.0, 1.0 - abs(x) / tol_x)
    score_y     = max(0.0, 1.0 - abs(y) / tol_y)
    score_vx    = max(0.0, 1.0 - abs(vx) / tol_vx)
    score_vy    = max(0.0, 1.0 - abs(vy) / tol_vy)
    score_angle = max(0.0, 1.0 - abs(angle) / tol_angle)
    score_leg   = (left_c + right_c) * 0.5   # linear in [0,1]

    soft_sum = (score_x + score_y + score_vx + score_vy +
                score_angle + score_leg)
    comp_soft = w_soft * soft_sum

    # ------------------------------------------------------------------
    # Component F - thrust penalty (fuel efficiency)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = (comp_prox + comp_vel + comp_angle +
                    comp_contact + comp_soft + comp_thrust)

    components = {
        "goal_proximity": comp_prox,
        "velocity_penalty": comp_vel,
        "angle_penalty": comp_angle,
        "contact_quality": comp_contact,
        "soft_success": comp_soft,
        "thrust_penalty": comp_thrust,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4.5. 历史最佳奖励函数代码
如果 Subagent 诊断建议回退到某个历史最佳版本，或你判断当前方向错误，应基于此最佳代码做最小修改，而不是从当前失败代码出发或全盘重建。
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation (using next_obs, which represents the state after action)
    x       = next_obs[0]   # horizontal offset from target pad center
    y       = next_obs[1]   # vertical offset from target pad (height)
    vx      = next_obs[2]   # horizontal velocity
    vy      = next_obs[3]   # vertical velocity
    angle   = next_obs[4]   # body tilt angle (rad)
    angvel  = next_obs[5]   # angular velocity
    left_c  = next_obs[6]   # left leg contact flag (0 or 1)
    right_c = next_obs[7]   # right leg contact flag (0 or 1)

    # ------------------------------------------------------------------
    # Component A – goal proximity (bounded, soft guidance)
    # ------------------------------------------------------------------
    # Changed from unbounded -abs(x)-abs(y) to a bounded penalty that
    # saturates far from the pad.
    w_prox  = 5.0
    dist_sq = x**2 + y**2
    comp_prox = w_prox * ( - dist_sq / (1.0 + dist_sq) )

    # ------------------------------------------------------------------
    # Component B – velocity penalty
    # ------------------------------------------------------------------
    w_vel = 1.0
    comp_vel = w_vel * ( -vx**2 - vy**2 )

    # ------------------------------------------------------------------
    # Component C – angle penalty
    # ------------------------------------------------------------------
    w_ang = 2.0
    comp_angle = w_ang * ( -angle**2 - angvel**2 )

    # ------------------------------------------------------------------
    # Component D – contact quality (stability constraint)
    # ------------------------------------------------------------------
    # Reward simultaneous leg contact, gated by position and velocity.
    w_contact = 10.0

    pos_thresh = 0.3
    vel_thresh = 0.3

    gate_x   = max(0.0, 1.0 - abs(x) / (pos_thresh + 1e-8))
    gate_y   = max(0.0, 1.0 - abs(y) / (pos_thresh + 1e-8))
    gate_vx  = max(0.0, 1.0 - abs(vx) / (vel_thresh + 1e-8))
    gate_vy  = max(0.0, 1.0 - abs(vy) / (vel_thresh + 1e-8))

    leg_contact = left_c * right_c
    comp_contact = w_contact * leg_contact * gate_x * gate_y * gate_vx * gate_vy

    # ------------------------------------------------------------------
    # Component E – success proxy (geometric mean of near‑perfect conditions)
    # ------------------------------------------------------------------
    w_proxy = 20.0

    eps_x     = 0.1
    eps_y     = 0.1
    eps_vx    = 0.15
    eps_vy    = 0.15
    eps_angle = 0.15

    score_x     = max(0.0, 1.0 - abs(x) / eps_x)
    score_y     = max(0.0, 1.0 - abs(y) / eps_y)
    score_vx    = max(0.0, 1.0 - abs(vx) / eps_vx)
    score_vy    = max(0.0, 1.0 - abs(vy) / eps_vy)
    score_angle = max(0.0, 1.0 - abs(angle) / eps_angle)
    score_contact = (left_c + right_c) * 0.5

    prod = (score_x * score_y * score_vx * score_vy * score_angle * score_contact)
    if prod > 1e-12:
        comp_proxy = w_proxy * (prod ** (1.0 / 6.0))
    else:
        comp_proxy = 0.0

    # ------------------------------------------------------------------
    # Component F – thrust penalty (efficiency)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = (comp_prox + comp_vel + comp_angle +
                    comp_contact + comp_proxy + comp_thrust)

    components = {
        "goal_proximity": comp_prox,
        "velocity_penalty": comp_vel,
        "angle_penalty": comp_angle,
        "contact_quality": comp_contact,
        "success_proxy": comp_proxy,
        "thrust_penalty": comp_thrust,
    }

    return float(total_reward), components
```

# 5. 组件逐项对比
（第一轮反思，无历史对比）

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-120.253372, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335058, -98.281266]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -156.891027 | -53.8% | 53.8% | 100.0% |
| velocity_penalty | -93.439696 | -32.0% | 32.0% | 99.8% |
| angle_penalty | -38.994126 | -13.4% | 13.4% | 100.0% |
| success_proxy | 1.434087 | 0.5% | 0.5% | 0.1% |
| contact_quality | 0.768669 | 0.3% | 0.3% | 0.2% |
| thrust_penalty | -0.182500 | -0.1% | 0.1% | 5.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主体从视野顶部中央附近以随机初始速度出发，目标是尽可能快速、稳定地降落在中央目标垫上并停稳（双腿接触、位置居中对齐、姿态水平、速度几乎为零）。次级目标是整个过程中尽量少使用发动机推力（省燃料）。不应混淆的目标：单纯的“存活”或“保持平衡”不是主目标，而是着陆成功的前提；单纯的“燃料为零”也不是主目标，只是在成功着陆前提下才受鼓励。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float64 或 float32（由环境决定）  
- obs[0]: name=x_position, meaning=相对于目标垫水平位置的偏差坐标，reward_usable: true  
- obs[1]: name=y_position, meaning=相对于目标垫高度的垂直偏差坐标，reward_usable: true  
- obs[2]: name=x_velocity, meaning=水平线速度，reward_usable: true  
- obs[3]: name=y_velocity, meaning=垂直线速度，reward_usable: true  
- obs[4]: name=body_angle, meaning=机体倾斜角度（弧度制），reward_usable: true  
- obs[5]: name=angular_velocity, meaning=角速度，reward_usable: true  
- obs[6]: name=left_support_contact, meaning=左支撑腿接触标志（1.0 表示接触），reward_usable: true  
- obs[7]: name=right_support_contact, meaning=右支撑腿接触标志（1.0 表示接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: name=no_engine, meaning=不点火，无推力  
- action 1: name=left_orientation_engine, meaning=点燃左姿态发动机（产生一个力矩和微小横向力）  
- action 2: name=main_engine, meaning=点燃主发动机（产生向上推力）  
- action 3: name=right_orientation_engine, meaning=点燃右姿态发动机（产生反向力矩和微小横向力）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  - body_not_awake_or_settled：机体进入睡眠或完全稳定不动。当双腿安全接触目标垫且速度、姿态均在阈值内时可能触发，属于成功信号（但无法直接与失败情形区分）。  
- failure-like termination:  
  - crash_or_body_contact：机体与地面或障碍猛烈碰撞（可能由于姿态过度倾斜或高速冲击非垫区域）→ 典型失败。  
  - horizontal_position_outside_viewport：水平位置超出有效视野范围 → 失败。  
- ambiguous termination:  
  - body_not_awake_or_settled 若发生在垫外（比如在倾斜坡道上稳定）也可能被终止，此时实际是失败，但终止模式相同。  
- truncation: 未在代码片段中明确给出最大步数，但通常存在隐含截断。本分析不依赖截断信号。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 字典为空，无 success 字段）  
- explicit_failure_flag_available: false （同上）  
- allowed_info_fields: 无。info 恒为空字典，不可用任何 info 键。  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用。

## 7. 可用于奖励函数的信号
- position: obs[0] (x 偏差), obs[1] (y 偏差) → 目标接近程度
- velocity: obs[2] (vx), obs[3] (vy) → 移动速度，用于鼓励减速
- orientation: obs[4] (body_angle) → 倾斜程度，应接近0
- angular_velocity: obs[5] → 角旋转速率，可辅助惩罚快速旋转
- contact: obs[6] (左腿接触), obs[7] (右腿接触) → 用于鼓励双腿着垫、避免单腿着陆
- action/engine: action 值本身 → 可惩罚非零动作（燃料使用）或针对性地惩罚姿态发动机频繁使用
- other: 无额外可用信号
```
