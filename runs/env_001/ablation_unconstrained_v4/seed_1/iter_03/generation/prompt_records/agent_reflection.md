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
# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: 函数体中使用了 original_reward (record: runs\env_001\ablation_unconstrained_v4\seed_1\iter_03\generation\validations\reward_v3.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
前一版奖励函数中 `proximity_shaping`（1/(1+dist)）提供了持续的正反馈，agent 因此学会在原点附近悬停而不着陆，导致 episode 长度极长（983）、几乎全是 truncation（18/20）、`contact_reward` 激活率极低（0.2%），且总 score 仍为负（-40.47）。需要去除这种“存在奖励”，强化进度信号和终端完成奖励，同时利用步惩罚和原始奖励抑制拖延。修改方案：

1. **移除 `proximity_shaping`**，切断悬停激励。
2. **增强距离进度**：`distance_progress` 系数从 3.0 升至 5.0。
3. **加入步惩罚**：每步 -0.05，鼓励快速完成任务。
4. **接触奖励改为无距离门控**：`contact_reward = 5.0 * contact_count`，仅靠接触本身引导。
5. **增加软着陆成功奖励**：当双腿接触、位置接近原点、速度/角度/角速度极低时，给予 +200，促使 agent 主动进入终止着陆状态。
6. **速度与姿态惩罚全局化**，删除 `near_pad` 门控，防止 agent 通过远离目标规避惩罚。
7. **纳入 `original_reward`**，使其直接参与总奖励，以利用环境自带的燃料消耗/坠毁惩罚。

以下为修改后的奖励函数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- extract current and next state ----
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- distances ----
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # ---- distance progress (scaled up) ----
    progress = prev_dist - next_dist
    c_progress = 5.0 * progress

    # ---- step penalty to discourage lingering ----
    c_step_penalty = -0.05

    # ---- contact reward (without proximity gate) ----
    contact_count = n_left_contact + n_right_contact  # 0.0, 1.0, 2.0
    c_contact = 5.0 * contact_count

    # ---- soft landing bonus (one large lump, will only trigger at termination) ----
    landed = (
        n_left_contact > 0.5 and n_right_contact > 0.5 and
        abs(nx) < 0.15 and abs(ny) < 0.15 and
        nvx**2 + nvy**2 < 0.04 and
        abs(n_angle) < 0.2 and
        abs(n_angvel) < 0.2
    )
    c_landing = 200.0 if landed else 0.0

    # ---- speed penalty (global) ----
    speed_sq = nvx**2 + nvy**2
    c_speed = -0.1 * speed_sq

    # ---- posture penalties (global) ----
    c_angle = -0.05 * (n_angle**2)
    c_angvel = -0.005 * (n_angvel**2)

    # ---- combine with original reward ----
    our_reward = (
        c_progress
        + c_step_penalty
        + c_contact
        + c_landing
        + c_speed
        + c_angle
        + c_angvel
    )
    total_reward = our_reward + original_reward

    components = {
        'distance_progress': c_progress,
        'step_penalty': c_step_penalty,
        'contact_reward': c_contact,
        'landing_bonus': c_landing,
        'speed_penalty': c_speed,
        'angle_penalty': c_angle,
        'angvel_penalty': c_angvel,
    }

    return (float(total_reward), components)
```

# Search objective
- target_score: 200.000000
- current_score: -40.472597
- gap_to_target: 240.472597
- target_achievement_ratio: -20.236%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -40.472597）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- extract current and next state ----
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- distances ----
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # proximity gate: ~1.0 near pad, decays to ~0 far away
    near_pad = 2.718281828 ** (-next_dist**2 / 0.5)

    # Component A: distance progress (strengthened main signal)
    progress = prev_dist - next_dist
    c_progress = 3.0 * progress

    # Component B: continuous proximity shaping (dense gradient everywhere)
    c_proximity = 1.0 / (1.0 + next_dist)

    # Component C: decomposed contact reward (additive, not product)
    contact_count = n_left_contact + n_right_contact  # [0.0, 1.0, 2.0]
    c_contact = 3.0 * contact_count * near_pad

    # Component D: speed penalty gated by proximity
    speed_sq = nvx**2 + nvy**2
    c_speed = -0.5 * speed_sq * near_pad

    # Component E: posture shaping (mild global + focused near-pad)
    c_angle_global = -0.03 * (n_angle**2)
    c_angle_near   = -0.5 * (n_angle**2) * near_pad
    c_angvel_global = -0.003 * (n_angvel**2)
    c_angvel_near   = -0.05 * (n_angvel**2) * near_pad

    total_reward = (
        c_progress
        + c_proximity
        + c_contact
        + c_speed
        + c_angle_global
        + c_angle_near
        + c_angvel_global
        + c_angvel_near
    )

    components = {
        'distance_progress': c_progress,
        'proximity_shaping': c_proximity,
        'contact_reward': c_contact,
        'speed_penalty': c_speed,
        'angle_global': c_angle_global,
        'angle_near': c_angle_near,
        'angvel_global': c_angvel_global,
        'angvel_near': c_angvel_near,
    }

    return (float(total_reward), components)
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-40.472597, len=983.300000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-152.586351, 10.424743]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_shaping | 677.191096 | 97.8% | 97.8% | 100.0% |
| contact_reward | 7.571002 | 1.1% | 1.1% | 0.2% |
| distance_progress | 3.344686 | 0.5% | 0.6% | 100.0% |
| speed_penalty | -3.163849 | -0.5% | 0.5% | 100.0% |
| angle_near | -0.430501 | -0.1% | 0.1% | 100.0% |
| angle_global | -0.085732 | -0.0% | 0.0% | 100.0% |
| angvel_near | -0.056521 | -0.0% | 0.0% | 100.0% |
| angvel_global | -0.006701 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个二维飞行器轨迹优化任务。智能体控制一个带有主引擎和左右姿态引擎的飞行器，从视口顶部中心附近出发。主目标是**以尽可能快的速度到达并稳定降落到中央目标垫上**，同时**尽可能少地使用引擎推力**（节省燃料）。次要目标包括保持姿态稳定、在接触时速度尽可能小（软着陆）、避免主体碰撞或飞出边界。  
**不应混淆的目标**：单纯追求速度而不计燃料，或不关心姿态稳定只求触碰目标垫。

## 3. 观察空间 observation_space
- **type:** `Box`
- **shape:** `[8]`
- **dtype:** `float32`（推测，因为标志位也以浮点数 1.0 / 0.0 表示）
- **各维度含义与奖励可用性：**

| 索引 | 名称 | 含义 | 奖励可用 |
|------|------|------|----------|
| 0 | `x_position` | 相对于目标垫中心的水平坐标，目标值为 0 | true |
| 1 | `y_position` | 相对于垫子高度的垂直坐标，目标值为 0 | true |
| 2 | `x_velocity` | 水平线速度 | true |
| 3 | `y_velocity` | 垂直线速度（正值向上，负值向下） | true |
| 4 | `body_angle` | 机体朝向角度 | true |
| 5 | `angular_velocity` | 角速度 | true |
| 6 | `left_support_contact` | 左支撑腿接触标志（1.0 或 0.0） | true |
| 7 | `right_support_contact` | 右支撑腿接触标志（1.0 或 0.0） | true |

## 4. 动作空间 action_space
- **type:** `Discrete`
- **n:** 4
- **动作含义：**

| 动作编号 | 名称 | 含义 |
|----------|------|------|
| 0 | `no_engine` | 什么都不做（无推力） |
| 1 | `left_orientation_engine` | 启动左姿态引擎（产生逆时针旋转力矩） |
| 2 | `main_engine` | 启动主引擎（产生向上的推力，抵消重力） |
| 3 | `right_orientation_engine` | 启动右姿态引擎（产生顺时针旋转力矩） |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination:**  
  由 `body_not_awake_or_settled` 触发，表示机体已稳定并且不再活跃。若此时满足着陆条件（x≈0, y≈0, 两腿接触, 速度和角速度极低），则可判定为成功着陆。  
- **failure-like termination:**  
  - `crash_or_body_contact`：机体主体或非腿部部分触地/碰到障碍物，视为坠毁。  
  - `horizontal_position_outside_viewport`：水平位置超出视口边界，视为飞出任务区域。  
- **ambiguous termination:**  
  - 理论上 `body_not_awake_or_settled` 若发生在非目标位置（如悬停不动但并未着陆），虽然罕见，但被视作失败（因为未完成到达目标垫的任务）。  
- **truncation:**  
  未提及，此处未使用。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available:** `false`（`info` 为空，无明确成功标志）
- **explicit_failure_flag_available:** `false`
- **allowed_info_fields:** 无（`info = {}`，任何字段均不可用）
- **forbidden_or_uncertain_info_fields:** `success`, `failure`, `termination_reason` 等任何未在 step 源码中出现的字段均禁止使用。

## 7. 可用于奖励函数的信号
- **位置信号:** `x_position` (goal=0), `y_position` (goal=0)  
- **速度信号:** `x_velocity`, `y_velocity`（希望着陆时接近 0）  
- **姿态与角速度:** `body_angle`, `angular_velocity`（着陆时应接近 0）  
- **接触信号:** `left_support_contact`, `right_support_contact`（两条腿都接触才满足软着陆）  
- **动作信号:** 离散动作编号，可用于推力消耗惩罚  
- **其他:** 可以从 `next_obs` 推断终止时的着陆成功（见 12 节）
```
