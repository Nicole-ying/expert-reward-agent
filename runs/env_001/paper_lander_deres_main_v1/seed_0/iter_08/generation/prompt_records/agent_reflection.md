# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的任务是先理解为什么当前的奖励函数失败了，再决定怎么修改。

# 先诊断，再行动

拿到训练反馈后，先回答三个问题：
0. 只根据下面提供的环境事实摘要理解任务、观测和动作，不得猜测或声称环境是某个已知 benchmark，环境身份对诊断没有必要。
1. 这个 agent 发生了什么？episode 很短（相对该环境正常长度明显偏短）→ 在 crash。episode 很长但得分差 → 在徘徊。得分已经好但某组件 ratio 异常 → 可能在 exploit。
2. 哪个组件是主要原因？不要只看 ratio，结合 nonzero_rate 和 episode_length 一起判断。
3. 我之前改了什么？从 Agent Memory 看上一轮的动作和效果。如果上次改了 A 但得分没变，这次不要再改 A。

**对齐检查：** `original_env_reward` 仅是未参与训练的参考信号。它的 mean 是每步平均值，不能仅凭数值大小推断“几乎没前进”、crash 或任务完成度；行为判断必须结合外部每回合 score、episode_length 和终止统计。`ratio_to_progress` 的符号只能作为潜在错位线索，不能单独证明 misalignment。

如果你不确定根因，用 search_reward_design_knowledge 查类似的失败模式。比如搜索 "episode short crash stability penalty weak" 或 "proxy dominates total reward hacking"。

# 工具

- search_reward_design_knowledge(query)：搜索设计技法库和失败模式库。当你对某个症状不确定原因或不知道怎么修时调用。
- get_skeleton_detail(skeleton_name)：查看某个骨架的数学形态、原理和陷阱。

# 怎么修订

三种层次，从轻到重：

**层次 1：改系数。** ratio_to_progress 判断组件间量级。惩罚项（stability_penalty、energy_penalty 等）ratio 绝对值 > 0.5 且外部得分差 → 考虑削弱。任务完成 proxy（soft_landing_proxy 等）ratio 天然偏高（经常 2~20），只要 nonzero_rate 正常（>5%）且外部得分不差，**不要因为它 ratio 高就削弱它**——削弱 proxy 会导致 agent 失去完成任务的唯一引导而 crash。bonus 类 ratio 天然偏高不是 bug。nonzero_rate < 2% → 增大权重或放宽条件。

**层次 2：改数学形式。** 同一个系数反复调还是不行，说明当前数学形式本身有问题。考虑改变信号的计算方式——但每次只改一个组件的形式，下一轮看效果。

**层次 3：换骨架。** 以下情况停止在层次 1/2 上打转，直接换主信号框架：
- 同一骨架家族已迭代 2 轮以上，且最佳得分仍未超过 target 的 25%。
- 或者已经改过数学形式（层次 2）但得分没有实质性改善。

**revert 规则：** 当 best_reward 得分明显高于 current 时，回到 best 的代码，但**必须在此基础上做一个新的修改**，不能原样复制。原样复制 = 浪费训练资源。例如 best 的 proxy 是 1.0，current 改成 0.15 崩了 → 回到 1.0 后换一个方向（如增强 progress、收紧 proxy 条件而非改系数），而不是删掉 0.15 就提交。

# 奖励函数迭代的通用原则

以下原则来自大量实验，与环境无关。

## 原则 1：比率是通用语言

不关心组件系数的绝对值。关心组件之间的相对大小和方向：
- 主学习信号应该是最强的正向力。
- 约束/惩罚（stability_penalty、energy_penalty 等负向信号）应该是弱背景信号——如果它的 ratio_to_progress 绝对值超过 0.5，agent 可能选择"不动"来避免惩罚，而非"行动"来获取奖励。
- 任务完成 proxy（soft_landing_proxy 等正向事件信号）的 ratio 天然偏大（经常 2~20），这不代表它有问题。agent 需要用这个信号学习怎么完成任务。只要 nonzero_rate 正常（>5%）且外部得分不差，高 ratio 不是 bug。**不要削弱正在工作的 proxy**——想提分应增强主信号或调整其他组件。
- `original_env_reward` 的 ratio 符号与主信号相反时，只标记为潜在 misalignment；必须结合外部每回合 score、episode_length 和终止统计验证后再决定是否 rebuild。

你从 feedback 的 ratio_to_progress 列能直接读到这些比率。

## 原则 2：数学形态决定梯度质量

- 二值条件 → 无梯度，触发率极低时等于摆设。
- 连续乘积 → 有梯度，但多个因子相乘会使整体信号很弱（每个因子 < 1）。
- 连续加权和 → 每个因子独立贡献梯度，但各因子之间没有"同时满足"的约束。
- bounded 函数（1/(1+kx)、max(0,1-x/D)、tanh）→ 自动限制值域，不受环境尺度影响。
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

先写注释说明诊断和修改理由，再输出完整 Python 代码。
函数签名：def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
返回 (float(total_reward), components)，components 只放总公式中直接出现的变量，不包含 total_reward。

```

## User Prompt

```markdown
# ⚠️ 上一版代码验证失败
错误信息：Reward v8 failed validation: runs\env_001\paper_lander_deres_main_v1\seed_0\iter_08\generation\validations\reward_v8.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -4345.464756）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for lunar-lander-style task (navigation_goal_reaching).
    
    Uses a potential-based shaping as the main progress signal and a soft
    landing proxy as a sparse success approximation.
    """
    
    # ----- Unpack observations -------------------------------------------------
    # Current
    x = obs[0]
    y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    left_contact = obs[6]
    right_contact = obs[7]

    # Next
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle = next_obs[4]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # ----- Derived quantities ---------------------------------------------------
    # Euclidean distance to target (origin)
    dist = (x ** 2.0 + y ** 2.0) ** 0.5
    next_dist = (next_x ** 2.0 + next_y ** 2.0) ** 0.5

    # Speed
    speed = (vx ** 2.0 + vy ** 2.0) ** 0.5
    next_speed = (next_vx ** 2.0 + next_vy ** 2.0) ** 0.5

    # ----- Component 1: Progress reward (potential-based shaping) ----------------
    # Phi(state) = -(dist + β_speed * speed + β_angle * |angle|)
    # The agent receives reward for decreasing distance, slowing down, and
    # aligning upright.
    beta_speed = 0.5
    beta_angle = 0.3
    gamma = 0.99

    phi = -(dist + beta_speed * speed + beta_angle * abs(angle))
    phi_next = -(next_dist + beta_speed * next_speed + beta_angle * abs(next_angle))

    progress_reward = gamma * phi_next - phi

    # ----- Component 2: Landing soft proxy (first touchdown) --------------------
    # Give a one-time bonus when both legs transition from no-contact to contact
    # while velocity and attitude are small (indicating a safe landing).
    landing_reward = 0.0
    landing_bonus = 10.0
    speed_threshold = 0.3
    angle_threshold = 0.1

    both_prev_air = (left_contact == 0.0) and (right_contact == 0.0)
    both_next_ground = (next_left == 1.0) and (next_right == 1.0)
    stable_touchdown = (next_speed < speed_threshold) and (abs(next_angle) < angle_threshold)

    if both_prev_air and both_next_ground and stable_touchdown:
        landing_reward = landing_bonus

    # ----- Total reward ---------------------------------------------------------
    total_reward = progress_reward + landing_reward

    # Components dict (only the terms that are summed)
    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next states
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]

    # 1. Core learning signal: progress towards target (0,0)
    dist_current = (x ** 2 + y ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_current - dist_next  # positive when approaching

    # 2. Continuous approach quality shaping
    #    Three bounded factors [0,1], product ensures all must be satisfied
    prox_factor = 1.0 / (1.0 + 5.0 * dist_next)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_angle))
    approach_quality = prox_factor * speed_factor * angle_factor  # [0, 1], dense

    # 3. Orientation penalty (gentle, keeps the craft upright)
    attitude_penalty = -0.01 * abs(next_angle)

    # Combine — w_approach reduced from 0.3 to 0.15 to curb hovering
    # (ratio was 280:1 vs progress, now targeting ~140:1 as first step)
    w_approach = 0.15
    approach_quality_reward = w_approach * approach_quality
    total_reward = progress + approach_quality_reward + attitude_penalty

    components = {
        "progress_reward": progress,
        "approach_quality_reward": approach_quality_reward,
        "attitude_penalty": attitude_penalty,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-4345.464756, len=383.100000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| landing_reward | 0.000120 | 0.000120 | 0.000012 | 10.000000 | 0.000831 |
| progress_reward | 0.143181 | 0.143849 | 1.000000 | 0.143181 | 0.995353 |
| total_reward | 0.143301 | 0.143969 | 1.000000 | 0.143301 | 0.996184 |
| original_env_reward | -5.786956 | 6.159514 | 1.000000 | -5.786956 | -40.229274 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_reward | 0.083102 | 0.083102 | 0.000000 | 80.000000 | 1444 |
| progress_reward | 99.159312 | 99.159312 | 0.744308 | 330.170053 | 1444 |
| total_reward | 99.242415 | 99.242415 | 0.744308 | 330.170053 | 1444 |

## Distribution
- score: mean=-4345.464756, min=-10799.475542, max=-395.312781
- episode_length: mean=383.100000
- early_terminal (<150 steps + score<-50): 6/20 (30%)
- errors: 0


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 飞行器轨迹优化任务。飞行器从视口顶部中央附近以随机初始速度出发，目标是尽可能快地到达并平稳降落在视口中央的指定着陆平台上，同时尽量少用引擎推力。智能体需要学会靠近目标点、降低速度、保持稳定姿态并实现安全接触（着陆腿与平台接触且最终静止）。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（连续值）
- obs[0]: x_position — 机体相对于目标平台中心的水平坐标
- obs[1]: y_position — 机体相对于平台高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体朝向角度
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑腿是否接触（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact — 右支撑腿是否接触（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: 无引擎（不做任何推进）
- action 1: 左姿态引擎（触发左侧姿态调整引擎）
- action 2: 主引擎（触发主推进引擎）
- action 3: 右姿态引擎（触发右侧姿态调整引擎）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - `body_not_awake_or_settled`：表示机体已经停止运动（低速、可能双脚着地且稳定），这通常是成功着陆的自然结果。

- failure-like termination:
  - `crash_or_body_contact`：机体与地面或障碍发生非允许的碰撞（如舱体直接撞击地面、侧翻等）。
  - `horizontal_position_outside_viewport`：机体水平位置超出边界，失去控制。

- ambiguous termination:
  - 无。

- truncation:
  - 环境未定义 episode 长度上限（通过其他方式截断），但通常会有一个最大步数限制作为安全截断，此处未明确给出。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （无可用字段，step 返回 info = {}）
- forbidden_or_uncertain_info_fields: 任何可能存在的 info 字段均不可用（如 success, failure, termination_reason 等）

## 7. 可用于奖励函数的信号
基于观测空间，可直接使用的信号包括：
- position: obs[0] 横向偏差、obs[1] 垂直偏差
- velocity: obs[2] 水平速度、obs[3] 垂直速度
- orientation: obs[4] 机体倾角、obs[5] 角速度
- contact: obs[6] 左腿接触标志、obs[7] 右腿接触标志
- action/engine: 动作 id（0-3）可用于惩罚引擎使用或鼓励特定策略
- 组合衍生信号：如是否双脚同时接地、速度是否接近零、位置是否接近 0（目标点）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | attitude_penalty + landing_quality_reward + progress_reward | -107.36 | -107.36 | 0.00 | 68.50 | attitude_penalty=-0.000 landing_quality_reward=0.005 progress_reward=0.016 | new_best |
| 2 | approach_quality_reward + attitude_penalty + progress_reward | 124.32 | 124.32 | 0.00 | 873.90 | approach_quality_reward=0.643 attitude_penalty=-0.000 progress_reward=0.002 | new_best |
| 3 | approach_quality_reward + attitude_penalty + progress_reward | 146.36 | 146.36 | 0.00 | 1000.00 | approach_quality_reward=0.094 attitude_penalty=-0.001 progress_reward=0.002 | new_best |
| 4 | approach_quality_reward + attitude_penalty + contact_bonus + progress_reward | 128.65 | 146.36 | -17.71 | 910.25 | approach_quality_reward=0.093 attitude_penalty=-0.000 contact_bonus=0.056 progress_reward=0.002 | no_meaningful_improvement |
| 5 | approach_quality_reward + attitude_penalty + landing_proxy + progress_reward | 139.74 | 146.36 | -6.62 | 1000.00 | approach_quality_reward=0.089 attitude_penalty=-0.001 landing_proxy=0.155 progress_reward=0.002 | no_meaningful_improvement |
| 6 | approach_quality_reward + attitude_penalty + progress_reward | 139.03 | 146.36 | -7.33 | 1000.00 | approach_quality_reward=0.063 attitude_penalty=-0.000 progress_reward=0.002 | unsolved_stagnation_fresh_restart |
| 7 | landing_reward + progress_reward | -4345.46 | 146.36 | -4491.83 | 383.10 | landing_reward=0.000 progress_reward=0.143 | no_meaningful_improvement |

```
