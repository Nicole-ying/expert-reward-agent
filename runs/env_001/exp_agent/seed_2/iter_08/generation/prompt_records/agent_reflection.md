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
# ⚠️ 上一版代码验证失败
错误信息：Reward v8 failed validation: runs\env_001\exp_agent\seed_2\iter_08\generation\validations\reward_v8.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 234.795902）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 读取当前状态
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 从 obs 读取上一步位置，计算距离变化
    px = obs[0]
    py = obs[1]

    prev_dist = (px**2 + py**2) ** 0.5
    curr_dist = (x**2 + y**2) ** 0.5

    # === 1. 主学习信号：进度增量（向目标靠近为正） ===
    # 这是 potential-based shaping 的特例：Φ = -distance, γ = 1
    # 正奖励 = 距离减小，天然零均值，提供清晰的梯度方向
    progress = prev_dist - curr_dist

    # === 2. 连续着陆质量信号（始终激活，峰值在着陆条件） ===
    # 三个 bounded 因子：proximity, speed, upright，都是 1/(1+kx) 形式
    # 自动 bounded 在 [0,1]，无需手动调尺度

    # 接近度：距离越近越高，k=5 使 distance=1 时约为 0.17
    proximity = 1.0 / (1.0 + 5.0 * curr_dist)

    # 速度因子：越慢越高，k=3 使 speed=1 时约为 0.25
    speed_val = (vx**2 + vy**2) ** 0.5
    speed_factor = 1.0 / (1.0 + 3.0 * speed_val)

    # 姿态因子：越正越高，k=3 使 angle=0.3rad 时约为 0.53
    upright_factor = 1.0 / (1.0 + 3.0 * abs(angle))

    # 接触加成：有腿部接触时额外奖励（0.5 到 1.0 之间）
    contact_bonus = 0.5 + 0.5 * (left_contact + right_contact) / 2.0

    # 连续乘积：每个因子 ∈ (0,1]，乘积提供密集梯度
    landing_quality = proximity * speed_factor * upright_factor * contact_bonus

    # === 3. 稳定性惩罚：距离门控，远处不罚 ===
    # 只在目标附近（<2 单位）施加弱稳定性约束
    gate = max(0.0, 1.0 - curr_dist / 2.0)
    stability_penalty = -gate * 0.02 * (abs(vx) + abs(vy) + abs(angular_vel))

    # === 组合 ===
    # progress: 方向性梯度，均值接近 0
    # landing_quality: 密集正信号，引导精细着陆行为
    # stability_penalty: 弱背景约束，仅在近端生效
    total_reward = progress + 1.0 * landing_quality + stability_penalty

    components = {
        "progress": progress,
        "landing_quality": landing_quality,
        "stability_penalty": stability_penalty,
        "total_reward": total_reward
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=234.795902, len=503.600000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| landing_quality | 0.485144 | 0.485144 | 1.000000 | 0.485144 |
| progress | 0.001971 | 0.002247 | 0.998118 | 0.001971 |
| stability_penalty | -0.003104 | 0.003104 | 0.999136 | -0.003104 |
| total_reward | 0.484011 | 0.484257 | 1.000000 | 0.484011 |
| generated_reward | 0.484011 | 0.484257 | 1.000000 | 0.484011 |
| original_env_reward | 0.033171 | 1.141502 | 1.000000 | 0.033171 |

## Distribution
- score: mean=234.795902, min=168.545600, max=284.504977
- episode_length: mean=503.600000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress + soft_landing_proxy + stability_penalty | -105.90 | -105.90 | 0.00 | 72.00 | progress=0.016 soft_landing_proxy=0.002 stability_penalty=-0.014 | new_best |
| 2 | progress + soft_landing_proxy + stability_penalty | 187.93 | 187.93 | 0.00 | 694.50 | progress=0.003 soft_landing_proxy=0.257 stability_penalty=-0.001 | new_best |
| 3 | progress + soft_landing_continuous + stability_penalty | 143.84 | 187.93 | -44.08 | 1000.00 | progress=0.003 soft_landing_continuous=0.237 stability_penalty=-0.001 | no_meaningful_improvement |
| 4 | progress + soft_landing_continuous + stability_penalty | 137.07 | 187.93 | -50.85 | 921.60 | progress=0.003 soft_landing_continuous=0.042 stability_penalty=-0.001 | no_meaningful_improvement |
| 5 | progress + soft_landing_proxy + stability_penalty | 144.59 | 187.93 | -43.34 | 1000.00 | progress=0.003 soft_landing_proxy=0.244 stability_penalty=-0.001 | unsolved_stagnation_fresh_restart |
| 6 | dist_reward + landing_proxy + stability_penalty | -113.31 | -113.31 | 0.00 | 71.90 | dist_reward=-0.972 landing_proxy=0.002 stability_penalty=-0.145 | new_best |
| 7 | landing_quality + progress + stability_penalty | 234.80 | 234.80 | 0.00 | 503.60 | landing_quality=0.485 progress=0.002 stability_penalty=-0.003 | target_solved_new_best |

```
