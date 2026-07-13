# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。用训练证据分析失败原因，提出修改方案并实施。你的目标不是匹配某个已知环境或骨架名称，而是改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- feedback来自训练后固定策略的同一批评估轨迹。`episode_sum_mean`表示每回合有符号累计量，`magnitude_share`表示绝对累计量份额，`signed_share`保留净方向，`active_rate`表示非零触发率。
- 组件统计是观察证据，不是因果贡献。必须结合score、episode_length、terminated/truncated、历史修改及其结果判断。
- 不同时间语义不可直接比较：逐步差分、持续状态值、惩罚和稀疏事件bonus不能套同一个比例阈值。
- 不得仅因任务描述出现"跳跃、着陆、抓取"等语义，就断言对应状态量是缺失奖励职责。新增职责必须有轨迹行为、终止分布、组件激活或历史干预结果支持；证据不足时明确写"未知"，优先保持best主结构并提出最小可证伪修改。
- episode达到时间上限且失败终止很少时，首先判断现有主信号是否已经实现稳定行为、剩余差距是否来自效率或主目标强度；没有行为证据时，不为动作过程本身添加proxy。

# 分析要求

拿到训练反馈后，回答：
1. 这个agent发生了什么？用行为证据说明。
2. 哪个组件最值得干预？结合数学形态和统计数据判断。
3. 我之前改了什么？从Agent Memory检查上一轮动作与实际效果。

随后根据分析结果修改奖励函数。你可以调整系数、改变数学形态、增加或删除组件、或重构整体结构——修改范围由证据决定。current明显差于best时，以best代码为基础，但必须做一个新的、有证据的修改，不能原样复制best。

# 工具

以下工具可在诊断不确定时调用：

- `search_reward_design_knowledge(query)`：检索相似失败模式和经验修复。
- `get_skeleton_detail(skeleton_name)`：查看数学形态、原理和陷阱。
- `get_reward_transformation(query)`：查看结构变换的原理、风险和验证目标。

# 代码约束

- 禁止terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的obs、next_obs、action和info字段，不得发明字段、切片维度或新输入。
- 第一个Python code block只能包含一个完整的`compute_reward`函数；不要写import、class、try/except或额外函数，不要使用self。
- 禁止eval/exec/open，禁止使用original_reward或原始环境reward。
- 需要平方根时使用`** 0.5`，禁止import numpy。需要指数形式时使用`2.718281828 ** exponent`，或改用`1/(1+k*x)`、`max(0,1-x/D)`等无需库的bounded表达式。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回`(float(total_reward), components)`；components只放总公式中直接出现的奖励组件，不放total_reward和中间调制器。

# 输出

先简要说明分析和修改理由，然后立即输出完整Python代码。

```

## User Prompt

```markdown
# Search objective
- target_score: 200.000000
- current_score: -121.198260
- gap_to_target: 321.198260
- target_achievement_ratio: -60.599%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -121.198260）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation to target platform.

    Key change from previous: settle_delta replaces absolute settle_reward.
    Settle quality improvement is rewarded per-step; holding steady gives zero.
    This prevents the "hover and farm" strategy seen with per-step absolute settle.
    """
    # Previous state
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    # Next state (result of action)
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speeds ---
    prev_speed = (pvx ** 2 + pvy ** 2) ** 0.5
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (tighter than before: k=7.0) ---
    k_prox = 7.0
    prev_prox = 1.0 / (1.0 + k_prox * prev_distance)
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # --- Settle quality factors (previous state) ---
    prev_velocity_factor = 1.0 / (1.0 + 3.0 * prev_speed)
    prev_angle_factor = 1.0 / (1.0 + 3.0 * abs(pangle))
    prev_contact_factor = 0.5 * (pleft_contact + pright_contact)
    prev_quality = prev_prox * (prev_velocity_factor + prev_angle_factor + prev_contact_factor)

    # --- Settle quality factors (current state) ---
    curr_velocity_factor = 1.0 / (1.0 + 3.0 * curr_speed)
    curr_angle_factor = 1.0 / (1.0 + 3.0 * abs(nangle))
    curr_contact_factor = 0.5 * (nleft_contact + nright_contact)
    curr_quality = curr_prox * (curr_velocity_factor + curr_angle_factor + curr_contact_factor)

    # ============================================================
    # Component A: progress_delta (primary approach guidance)
    # ============================================================
    w_progress = 3.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (steady baseline pull)
    # ============================================================
    w_dist = 0.5
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: velocity_damping (proximity-gated speed penalty)
    # ============================================================
    w_vel = 1.5
    velocity_damping = -w_vel * curr_prox * curr_speed

    # ============================================================
    # Component D: settle_delta (improvement in landing quality)
    # ============================================================
    w_settle = 8.0
    settle_delta = w_settle * (curr_quality - prev_quality)

    # ============================================================
    # Component E: orientation_penalty (proximity-gated tilt penalty)
    # ============================================================
    w_orient = 0.5
    w_angvel = 0.15
    orientation_penalty = -w_orient * curr_prox * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component F: engine_penalty (fuel efficiency)
    # ============================================================
    w_engine = 0.05
    engine_penalty = -w_engine if action != 0 else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + velocity_damping
        + settle_delta
        + orientation_penalty
        + engine_penalty
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "velocity_damping": float(velocity_damping),
        "settle_delta": float(settle_delta),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-121.198260, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-145.325878, -96.939078]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_penalty | -33.301118 | -52.2% | 52.2% | 100.0% |
| velocity_damping | -20.208528 | -31.7% | 31.7% | 100.0% |
| settle_delta | 5.716837 | 9.0% | 9.5% | 100.0% |
| progress_delta | 3.344923 | 5.2% | 5.4% | 100.0% |
| orientation_penalty | -0.636268 | -1.0% | 1.0% | 100.0% |
| engine_penalty | -0.182500 | -0.3% | 0.3% | 5.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是 **2D 飞行器着陆控制**：一个搭载主推进器与姿态推进器的刚体从视口顶部附近出发，必须在最短时间内飞向并稳定停靠在画面中央的目标平台上，且尽可能少地使用引擎推力。  
主要目标：**到达目标位置并安全、稳定地着陆**（接近目标、减小速度、保持直立姿态、双腿接触平台）。  
次要目标：耗费更少的燃料（即减少不必要的引擎动作），并尽可能快地完成着陆。  
不该混淆的目标：纯粹的速度最小化或仅追求时间最短而不考虑着陆稳定性。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: x_position – 相对于目标平台的水平坐标，reward_usable: true
- obs[1]: y_position – 相对于目标平台高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity – 水平线速度，reward_usable: true
- obs[3]: y_velocity – 垂直线速度，reward_usable: true
- obs[4]: body_angle – 机体倾斜角度，reward_usable: true
- obs[5]: angular_velocity – 角速度，reward_usable: true
- obs[6]: left_support_contact – 左支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true
- obs[7]: right_support_contact – 右支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action/action_dim 0: no_engine（无任何引擎推力）
- action/action_dim 1: left_orientation_engine（启动左侧姿态引擎，施加旋转冲量）
- action/action_dim 2: main_engine（启动主引擎，产生纵向推力）
- action/action_dim 3: right_orientation_engine（启动右侧姿态引擎，施加反向旋转冲量）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination:** `body_not_awake_or_settled`（机体停止运动/休眠，通常意味着已经稳定着陆在目标平台上）。
- **failure-like termination:** `crash_or_body_contact`（发生碰撞或除双腿外其他部位接触地面），`horizontal_position_outside_viewport`（水平位置飞出视口边界）。
- **ambiguous termination:** 无。
- **truncation:** 源码未显式提供截断，但环境可能带有默认的最大回合步数，其信息未在给出的 step 源码中体现，暂视为不存在。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 当前 step 源码返回空字典 `{}`
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（不存在）

## 7. 可用于奖励函数的信号
- **position:** x_position, y_position（均为相对于目标的坐标，接近零时表示已到达）
- **velocity:** x_velocity, y_velocity（用于抑制速度，特别是在着陆阶段）
- **orientation:** body_angle, angular_velocity（用于抑制滚动与倾斜）
- **contact:** left_support_contact, right_support_contact（标志是否已安全落在平台上）
- **action/engine:** action index（可区分是否启动引擎、哪一个引擎，以便进行燃料惩罚）
- **other:** 可从组合状态中提取“着陆成功”的复合特征（位置近零、速度极小、角度极小、双腿均接触）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | engine_efficiency + orientation_penalty + progress_to_goal + successful_settle_proxy | -121.66 | -121.66 | 0.00 | 68.35 | engine_efficiency=-0.014 orientation_penalty=-0.020 progress_to_goal=-0.972 successful_settle_proxy=0.035 | new_best |
| 2 | distance_penalty + engine_penalty + orientation_penalty + progress_delta + settle_reward + velocity_damping | 108.12 | 108.12 | 0.00 | 743.30 | distance_penalty=-0.127 engine_penalty=-0.033 orientation_penalty=-0.004 progress_delta=0.003 settle_reward=2.290 | new_best |
| 3 | distance_penalty + engine_penalty + orientation_penalty + progress_delta + settle_delta + velocity_damping | -121.20 | 108.12 | -229.31 | 68.40 | distance_penalty=-0.486 engine_penalty=-0.009 orientation_penalty=-0.008 progress_delta=0.049 settle_delta=0.091 | no_meaningful_improvement |

```
