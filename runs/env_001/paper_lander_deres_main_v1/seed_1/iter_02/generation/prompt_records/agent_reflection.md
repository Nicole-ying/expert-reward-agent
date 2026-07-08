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
# 上一轮奖励函数代码（该轮得分: 241.414498）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === Progress reward (main learning signal) ===
    x_prev, y_prev = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress_reward = dist_prev - dist_next  # positive when getting closer

    # === Landing quality bonus (soft proxy for successful landing) ===
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)

    landing_quality_bonus = 0.0
    if both_contact:
        # position error from center of the pad
        pos_err = dist_next
        # velocity magnitude
        vx, vy = next_obs[2], next_obs[3]
        vel = (vx ** 2 + vy ** 2) ** 0.5
        # absolute body angle
        angle = abs(next_obs[4])

        # exponential-based quality score (max ~1.0 for perfect landing)
        # temperatures control how quickly the score decays with imperfection
        temp_pos = 0.2
        temp_vel = 0.5
        temp_angle = 0.1
        quality = (2.718281828 ** (-pos_err / temp_pos)) * \
                  (2.718281828 ** (-vel / temp_vel)) * \
                  (2.718281828 ** (-angle / temp_angle))
        landing_quality_bonus = quality

    total_reward = progress_reward + landing_quality_bonus

    components = {
        "progress_reward": progress_reward,
        "landing_quality_bonus": landing_quality_bonus
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=241.414498, len=394.000000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| landing_quality_bonus | 0.369599 | 0.369599 | 0.541875 | 0.682074 | 89.453967 |
| progress_reward | 0.003847 | 0.004132 | 0.994395 | 0.003868 | 0.931035 |
| total_reward | 0.373445 | 0.373656 | 0.999976 | 0.373454 | 90.385002 |
| original_env_reward | -0.128733 | 1.590839 | 1.000000 | -0.128733 | -31.157284 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_quality_bonus | 121.322329 | 121.322329 | 0.000000 | 817.357410 | 3047 |
| progress_reward | 1.265536 | 1.265536 | 0.329691 | 1.420877 | 3047 |
| total_reward | 122.587865 | 122.587865 | 0.329691 | 818.757685 | 3047 |

## Distribution
- score: mean=241.414498, min=81.923051, max=298.037395
- episode_length: mean=394.000000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个2D飞行器/着陆器轨迹优化任务。飞行器初始位于画面顶部中央附近，带有随机初始作用力。  
**核心目标**：飞行器应尽快且平稳地降落在场地中央的“目标垫”上，并保持稳定的姿态与相对静止。  
**附属约束**：尽可能少地使用引擎推力，但这不是与核心目标冲突的多目标优化任务，而是围绕“高效着陆”的自然偏好。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: 推断为 float (位置、速度、角度、角速度为连续值，接触标志用 1.0/0.0 表示)  
- obs[0]: x_position — 飞行器相对于目标垫中心的水平距离 (接近0表示水平对齐)  
- obs[1]: y_position — 飞行器相对于目标垫高度的垂直距离 (接近0表示正好在垫面高度)  
- obs[2]: x_velocity — 水平线速度  
- obs[3]: y_velocity — 垂直线速度  
- obs[4]: body_angle — 机体倾斜角 (接近0表示竖直)  
- obs[5]: angular_velocity — 角速度  
- obs[6]: left_support_contact — 左支撑脚触地标志 (1.0 触碰, 0.0 未触碰)  
- obs[7]: right_support_contact — 右支撑脚触地标志 (1.0 触碰, 0.0 未触碰)

## 4. 动作空间 action_space
- type: Discrete (4)  
- action 0: no_engine — 不点火，无推力输出  
- action 1: left_orientation_engine — 侧向引擎喷火，产生侧向推力（用于调整机头朝向/水平速度）  
- action 2: main_engine — 主引擎喷火，产生向上的主要推力（用于减速/悬停）  
- action 3: right_orientation_engine — 与左侧引擎对称的侧向喷火

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` 触发时，若飞行器双支撑脚均触地 (`left_support_contact == 1.0` 且 `right_support_contact == 1.0`)、水平/垂直位置接近 0、速度很小、机体角度近乎竖直，则很可能是一次成功着陆。  
- failure-like termination:  
  - `crash_or_body_contact` — 机体与地面非目标区域剧烈接触（如翻倒、撞击地面），通常视为失败。  
  - `horizontal_position_outside_viewport` — 飞出水平边界，视为失败。  
- ambiguous termination:  
  `body_not_awake_or_settled` 也可能出现在非成功场景（例如飞行器靠某种方式静止在错误位置），因此单靠终止信号无法完全确定成功与否，需要结合观察信号自行判定。  
- truncation: 源信息中未提及 episode 步长上限，因此无显式 truncation；但实际使用中可能存在外部限制（如最大步数），非环境内建。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: (空) — 该环境 `info` 返回空字典 `{}`，因此 `info` 不提供任何额外字段。  
- forbidden_or_uncertain_info_fields: `info` 所有字段均不可用，`info["success"]`、`info["termination_reason"]` 等均不存在。

## 7. 可用于奖励函数的信号
从 `next_obs` 中可直接提取的、可量化的信号包括：
- **位置**：`next_obs[0]` (x), `next_obs[1]` (y) — 表示与目标垫的相对距离，可用于塑造接近/居中的奖励。  
- **速度**：`next_obs[2]` (vx), `next_obs[3]` (vy) — 可用于惩罚过高触地速度。  
- **姿态**：`next_obs[4]` (角度) — 可用于奖励竖直姿态（接近0）。  
- **角速度**：`next_obs[5]` — 可用于惩罚快速旋转。  
- **接触状态**：`next_obs[6]`, `next_obs[7]` — 双脚同时触地可作为成功着陆的有力判据，也可用于塑造中间奖励。  
- **动作/引擎使用**：`action` 为非零（使用引擎）时，可考虑付出燃油成本。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_quality_bonus + progress_reward | 241.41 | 241.41 | 0.00 | 394.00 | landing_quality_bonus=0.370 progress_reward=0.004 | target_solved_new_best |

```
