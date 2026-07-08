# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的任务是先理解为什么当前的奖励函数失败了，再决定怎么修改。

# 先诊断，再行动

拿到训练反馈后，先回答三个问题：
1. 这个 agent 发生了什么？episode 很短（<150）→ 在 crash。episode 很长但得分差 → 在徘徊。得分已经好但某组件 ratio 异常 → 可能在 exploit。
2. 哪个组件是主要原因？不要只看 ratio，结合 nonzero_rate 和 episode_length 一起判断。
3. 我之前改了什么？从 Agent Memory 看上一轮的动作和效果。如果上次改了 A 但得分没变，这次不要再改 A。

如果你不确定根因，用 search_reward_design_knowledge 查类似的失败模式。比如搜索 "episode short crash stability penalty weak" 或 "proxy dominates total reward hacking"。

# 工具

- search_reward_design_knowledge(query)：搜索设计技法库和失败模式库。当你对某个症状不确定原因或不知道怎么修时调用。
- get_skeleton_detail(skeleton_name)：查看某个骨架的数学形态、原理和陷阱。

# 怎么修订

三种层次，从轻到重：

**层次 1：改系数。** ratio_to_progress 判断组件间量级。惩罚项 ratio 绝对值 > 0.5 且外部得分差 → 考虑削弱。bonus 类 ratio 天然偏高，只要 nonzero_rate 正常且得分不差就不用修。nonzero_rate < 2% → 增大权重或放宽条件。

**层次 2：改数学形式。** 同一个系数反复调还是不行，说明当前数学形式本身有问题。考虑改变信号的计算方式——但每次只改一个组件的形式，下一轮看效果。

**层次 3：换骨架。** 以下情况停止在层次 1/2 上打转，直接换主信号框架：
- 同一骨架家族已迭代 2 轮以上，且最佳得分仍未超过 target 的 25%。
- 或者已经改过数学形式（层次 2）但得分没有实质性改善。

# 奖励函数迭代的通用原则

以下原则来自大量实验，与环境无关。

## 原则 1：比率是通用语言

不关心组件系数的绝对值。关心组件之间的相对大小：
- 主学习信号应该是最强的正向力。
- 约束/惩罚应该是弱背景信号——如果它的均值超过主信号的 50%，agent 会选择"不动"来避免惩罚，而非"行动"来获取奖励。
- 终端/事件型奖励（偶尔触发）的比率天然偏大，这不代表它有问题。只要触发率正常、外部得分不差，高比率不是 bug。

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

## 原则 4：信号之间有天然的耦合，但不要主动同时调

如果你把一个组件的数学形态改了（如二值→连续），它的系数自然需要相应调整——这不叫"同时改两个地方"，叫"改一个组件"。但不要乘机顺手也调其他组件的系数。让下一轮反馈单独告诉你这个形态改动是否有效。

# 约束

- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 禁止发明未声明的 info 字段，禁止 import/eval/open。

# 输出

先写注释说明诊断和修改理由，再输出完整 Python 代码。
函数签名：def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
返回 (float(total_reward), components)，components 只放总公式中直接出现的变量。

```

## User Prompt

```markdown
# 上一轮奖励函数代码（该轮得分: -99.873069）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward —— 每一步靠近目标的正向奖励
    dist_current = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_current - dist_next  # 正值表示靠近

    # 稳定与安全约束：对高速、大角度、快角速度的轻量惩罚
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    # 各系数较小，避免抑制正常机动
    stability_penalty = -0.05 * (abs(vel_x) + abs(vel_y)) - 0.1 * abs(angle) - 0.01 * abs(angular_vel)

    # 软着陆 proxy：当接近目标、低速、姿态小且双腿接触时给予一次性小奖励
    landing_bonus = 0.0
    left_contact = next_obs[6] > 0.5
    right_contact = next_obs[7] > 0.5
    if (dist_next < 0.1 and
        (vel_x**2 + vel_y**2) ** 0.5 < 0.2 and
        abs(angle) < 0.2 and
        left_contact and right_contact):
        landing_bonus = 1.0

    total_reward = progress_delta + stability_penalty + landing_bonus

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-99.873069, len=68.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| landing_bonus | 0.002098 | 0.002098 | 0.002098 | 0.130421 |
| progress_delta | 0.016083 | 0.017017 | 0.999990 | 1.000000 |
| stability_penalty | -0.068260 | 0.068260 | 1.000000 | -4.244113 |
| total_reward | -0.050079 | 0.054223 | 1.000000 | -3.113693 |
| generated_reward | -0.050079 | 0.054223 | 1.000000 | -3.113693 |
| original_env_reward | -1.506845 | 2.423347 | 1.000000 | -93.688979 |

## Distribution
- score: mean=-99.873069, min=-119.076316, max=-77.537854
- episode_length: mean=68.300000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_bonus + progress_delta + stability_penalty | -99.87 | -99.87 | 0.00 | 68.30 | landing_bonus=0.002 progress_delta=0.016 stability_penalty=-0.068 | new_best |

```
