# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的任务是先理解为什么当前的奖励函数失败了，再决定怎么修改。

# 先诊断，再行动

拿到训练反馈后，先回答三个问题：
1. 这个 agent 发生了什么？episode 很短（相对该环境正常长度明显偏短）→ 在 crash。episode 很长但得分差 → 在徘徊。得分已经好但某组件 ratio 异常 → 可能在 exploit。
2. 哪个组件是主要原因？不要只看 ratio，结合 nonzero_rate 和 episode_length 一起判断。
3. 我之前改了什么？从 Agent Memory 看上一轮的动作和效果。如果上次改了 A 但得分没变，这次不要再改 A。

**对齐检查：** 看 `original_env_reward` 的 `ratio_to_progress` 符号。如果和主信号符号相反 → 奖励函数在给失败行为发正反馈 → misaligned，需要 rebuild 而不是 tune。

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
- `original_env_reward` 的 ratio 如果与主信号符号相反（一正一负）→ 奖励函数在给失败行为发正反馈 → misaligned，需要 rebuild。

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
返回 (float(total_reward), components)，components 只放总公式中直接出现的变量，不包含 total_reward。

```

## User Prompt

```markdown
# ⚠️ 上一版代码验证失败
错误信息：Reward v8 failed validation: runs\env_001\exp_v8\seed_0\iter_08\generation\validations\reward_v8.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 137.606613）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 位置
    cx, cy = float(obs[0]), float(obs[1])
    nx, ny = float(next_obs[0]), float(next_obs[1])

    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：逐步靠近目标（scale=8，符合 skeleton 推荐 5~20）
    progress_reward = (d_curr - d_next) * 8.0

    # 2. 轻量稳定约束
    vx, vy = float(next_obs[2]), float(next_obs[3])
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(float(next_obs[4]))
    angular_v = abs(float(next_obs[5]))

    raw_stability_penalty = (
        -0.005 * speed
        - 0.01 * angle
        - 0.005 * angular_v
    )

    # stage weighting: 早期忽略 stability，后期逐渐加入
    # t=0 → late_weight=0（无惩罚）；t=1 → late_weight=1（全惩罚）
    t = training_progress
    late_weight = max(0.0, min(1.0, 2.0 * t - 0.5))
    stability_penalty = raw_stability_penalty * late_weight

    # 3. 软着陆近似信号：连续乘积因子 + x/(1+x) 饱和
    #    相比裸乘积（ratio 6.9x），饱和形式将输出压缩到 [0, 0.5]
    #    保留低值梯度（导数在 0 处≈1），压制高值 exploit 空间
    proximity_factor = max(0.0, 1.0 - d_next / 0.3)
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - angle / 0.3)
    contact_factor = (
        max(0.0, float(next_obs[6])) * max(0.0, float(next_obs[7]))
    )

    raw_proxy = proximity_factor * speed_factor * angle_factor * contact_factor
    # x/(1+x) 饱和：raw_proxy ∈ [0,1] → soft_landing_proxy ∈ [0, 0.5]
    soft_landing_proxy = raw_proxy / (1.0 + raw_proxy)

    # 总奖励
    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward,
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=137.606613, len=1000.000000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| progress_reward | 0.034361 | 0.036802 | 0.999804 | 1.000000 |
| soft_landing_proxy | 0.210295 | 0.210295 | 0.481814 | 6.120186 |
| stability_penalty | -0.000000 | 0.000000 | 0.003508 | -0.000000 |
| total_reward | 0.244655 | 0.246697 | 0.999993 | 7.120186 |
| generated_reward | 0.244655 | 0.246697 | 0.999993 | 7.120186 |
| original_env_reward | -0.247575 | 1.769447 | 1.000000 | -7.205159 |
| original_env_reward | -0.247575 | 1.769447 | 1.000000 | -7.205159 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=137.606613, min=107.956019, max=171.250968
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_proxy + stability_penalty | -88.79 | 0.00 | -88.79 | 69.00 | progress_reward=0.016 soft_landing_proxy=0.003 stability_penalty=-0.058 | new_best |
| 2 | progress_reward + soft_landing_proxy + stability_penalty | -108.77 | -88.79 | -19.98 | 68.45 | progress_reward=0.016 soft_landing_proxy=0.003 stability_penalty=-0.006 | no_meaningful_improvement |
| 3 | progress_reward + soft_landing_proxy + stability_penalty | 169.76 | 169.76 | 0.00 | 922.35 | progress_reward=0.043 soft_landing_proxy=0.294 stability_penalty=-0.000 | new_best |
| 4 | progress_reward + soft_landing_proxy + stability_penalty | -220.92 | 169.76 | -390.68 | 74.70 | progress_reward=0.136 soft_landing_proxy=0.000 stability_penalty=-0.000 | no_meaningful_improvement |
| 5 | progress_reward + soft_landing_proxy + stability_penalty | 169.76 | 169.76 | 0.00 | 922.35 | progress_reward=0.043 soft_landing_proxy=0.294 stability_penalty=-0.000 | no_meaningful_improvement |
| 6 | progress_reward + soft_landing_proxy + stability_penalty | -251.56 | 169.76 | -421.31 | 73.30 | progress_reward=0.140 soft_landing_proxy=0.000 stability_penalty=-0.000 | unsolved_stagnation_fresh_restart |
| 7 | progress_reward + soft_landing_proxy + stability_penalty | 137.61 | 137.61 | 0.00 | 1000.00 | progress_reward=0.034 soft_landing_proxy=0.210 stability_penalty=-0.000 | new_best |

```
