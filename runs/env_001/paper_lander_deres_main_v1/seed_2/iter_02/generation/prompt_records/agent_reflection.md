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
# 上一轮奖励函数代码（该轮得分: -161.078864）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v1 for lunar-lander-like 2D flying task.
    Components:
    1. progress_reward:   -distance_to_landing_pad (dense guidance)
    2. orientation_penalty: small penalty for non‑zero body angle and angular velocity
    3. landing_bonus:       large positive reward when lander is close, slow, and touching ground
    """
    # ---- unpack observations ----
    # current state
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    # left_contact = obs[6]   # not needed for progress
    # right_contact = obs[7]

    # next state (for landing condition)
    n_x_pos = next_obs[0]
    n_y_pos = next_obs[1]
    n_x_vel = next_obs[2]
    n_y_vel = next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- 1. progress_reward: negative Euclidean distance to target ----
    distance_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    progress_reward = -distance_current

    # ---- 2. orientation_penalty: encourage upright and still attitude ----
    # Small coefficients so it doesn't dominate the learning signal.
    angle_coeff = 0.05
    angvel_coeff = 0.01
    orientation_penalty = -angle_coeff * abs(body_angle) - angvel_coeff * abs(angular_vel)

    # ---- 3. landing_bonus: soft proxy for successful landing ----
    # Trigger conditions:
    #   - close to the pad centre
    #   - low total speed
    #   - at least one leg in contact with ground
    close_thresh = 0.3
    speed_thresh = 0.5
    bonus_value = 10.0

    n_distance = (n_x_pos ** 2 + n_y_pos ** 2) ** 0.5
    n_speed = (n_x_vel ** 2 + n_y_vel ** 2) ** 0.5
    legs_contact = (n_left_contact > 0.5) or (n_right_contact > 0.5)

    landing_bonus = 0.0
    if n_distance < close_thresh and n_speed < speed_thresh and legs_contact:
        landing_bonus = bonus_value

    # ---- total reward ----
    total_reward = progress_reward + orientation_penalty + landing_bonus

    # components dict (only includes the terms that are summed)
    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-161.078864, len=77.850000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| landing_bonus | 0.254424 | 0.254424 | 0.025442 | 10.000000 | 0.269106 |
| orientation_penalty | -0.020067 | 0.020067 | 1.000000 | -0.020067 | -0.021225 |
| progress_reward | -0.945442 | 0.945442 | 1.000000 | -0.945442 | -1.000000 |
| total_reward | -0.711085 | 1.210921 | 1.000000 | -0.711085 | -0.752119 |
| original_env_reward | -2.215704 | 4.003521 | 1.000000 | -2.215704 | -2.343564 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_bonus | 19.001265 | 19.001265 | 0.000000 | 7370.000000 | 13437 |
| orientation_penalty | -1.498547 | 1.498547 | -12.503569 | -0.047418 | 13437 |
| progress_reward | -70.595356 | 70.595356 | -521.598896 | -41.901344 | 13437 |
| total_reward | -53.092638 | 69.965459 | -530.999269 | 7177.928743 | 13437 |

## Distribution
- score: mean=-161.078864, min=-299.212645, max=-5.792269
- episode_length: mean=77.850000
- early_terminal (<150 steps + score<-50): 16/20 (80%)
- errors: 0


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
一个 2D 飞行器从画面顶部中央出发，受随机初始力影响。核心目标是让飞行器尽快到达画面中央的目标着陆垫并稳定停稳，同时尽量减少引擎推力使用。智能体需要学习靠近目标、减速、保持姿态稳定并实现安全着陆。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: 飞行器相对于目标垫中心的水平坐标 (x_position)
- obs[1]: 飞行器相对于目标垫高度的垂直坐标 (y_position)
- obs[2]: 水平线速度 (x_velocity)
- obs[3]: 垂直线速度 (y_velocity)
- obs[4]: 机身姿态角 (body_angle)
- obs[5]: 角速度 (angular_velocity)
- obs[6]: 左支撑腿接触标志，接触为 1.0，否则为 0.0 (left_support_contact)
- obs[7]: 右支撑腿接触标志，接触为 1.0，否则为 0.0 (right_support_contact)

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: 无引擎推力 (no_engine)
- action 1: 点燃左侧姿态引擎 (left_orientation_engine)
- action 2: 点燃主引擎 (main_engine)
- action 3: 点燃右侧姿态引擎 (right_orientation_engine)

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（飞行器停止活动并稳定，可能表示成功着陆并停稳）
- failure-like termination: crash_or_body_contact（撞击或非正常身体接触），horizontal_position_outside_viewport（水平出界）
- ambiguous termination: body_not_awake_or_settled 在未接触目标垫或错误位置时可能不表示真正成功
- truncation: 未明确提及，假设无时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （无，info为空字典）
- forbidden_or_uncertain_info_fields: info（空字典，不可提供任何额外信号）

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])，可用于衡量与目标垫的距离
- velocity: x_velocity (obs[2]), y_velocity (obs[3])，可用于衡量降落时的速度大小
- orientation: body_angle (obs[4]), angular_velocity (obs[5])，可用于姿态稳定
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])，可用于判断着陆
- action/engine: 动作选择本身（主引擎或姿态引擎），可用于惩罚推力使用

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_bonus + orientation_penalty + progress_reward | -161.08 | -161.08 | 0.00 | 77.85 | landing_bonus=0.254 orientation_penalty=-0.020 progress_reward=-0.945 | new_best |

```
