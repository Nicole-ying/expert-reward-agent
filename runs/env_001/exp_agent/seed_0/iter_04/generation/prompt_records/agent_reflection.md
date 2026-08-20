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
错误信息：Reward v4 failed validation: runs\env_001\exp_agent\seed_0\iter_04\generation\validations\reward_v4.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -1771.110673）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 提取当前状态 ──
    x, y = obs[0], obs[1]
    curr_vx, curr_vy = obs[2], obs[3]
    curr_angle = obs[4]

    # ── 提取下一时刻状态（反映动作后果）──
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 距离 ──
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5

    # ── 速率标量 ──
    speed = (vx**2 + vy**2) ** 0.5
    curr_speed = (curr_vx**2 + curr_vy**2) ** 0.5

    # ── 主学习信号：potential-based shaping ──
    # Φ = -(distance + α*speed + β*|angle|)
    # F = γ * Φ(next) - Φ(curr)，天然引导接近+减速+姿态稳定
    alpha = 0.1
    beta = 0.2
    gamma = 0.99

    phi_curr = -(dist + alpha * curr_speed + beta * abs(curr_angle))
    phi_next = -(next_dist + alpha * speed + beta * abs(angle))
    potential_shaping = gamma * phi_next - phi_curr

    # ── 软着陆 proxy（连续乘积，阈值放宽）──
    # 用 bounded max(0, 1-x/threshold) 提供连续梯度
    prox_factor = max(0.0, 1.0 - next_dist / 1.0)               # 距离 < 1.0
    vel_factor = max(0.0, 1.0 - speed / 0.5)                    # 速率 < 0.5
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.3)             # 倾角 < 0.3
    ang_vel_factor = max(0.0, 1.0 - abs(ang_vel) / 0.3)         # 角速度 < 0.3
    contact_factor = min(left_contact, right_contact)            # 双脚触地

    soft_landing_proxy = (
        prox_factor * vel_factor * angle_factor * ang_vel_factor * contact_factor
    )

    # ── 轻量辅助惩罚 ──
    # 角速度惩罚：抑制剧烈旋转
    angular_vel_penalty = -0.01 * abs(ang_vel)
    # 能量惩罚：鼓励高效移动
    energy_penalty = -0.01 * (abs(vx) + abs(vy))

    # ── 总奖励 ──
    total_reward = (
        1.0 * potential_shaping +
        2.0 * soft_landing_proxy +
        1.0 * angular_vel_penalty +
        1.0 * energy_penalty
    )

    components = {
        "potential_shaping": potential_shaping,
        "soft_landing_proxy": soft_landing_proxy,
        "angular_vel_penalty": angular_vel_penalty,
        "energy_penalty": energy_penalty,
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取当前和下一时刻的位置
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    # 速度、姿态、接触信息（使用 next_obs 更合理，反映动作导致的后果）
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 主学习信号：进度增量奖励 ----------
    # 每一步奖励“当前到目标的距离”与“下一步到目标的距离”之差
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5
    progress_delta = dist - next_dist   # 正值表示更靠近目标

    # ---------- 稳定/安全惩罚 ----------
    # 惩罚水平、垂直速度，以及姿态角和角速度，鼓励稳定接近
    stability_penalty = -(
        0.1 * abs(vx) +
        0.1 * abs(vy) +
        0.2 * abs(angle) +
        0.1 * abs(ang_vel)
    )

    # ---------- 任务完成近似信号（软着陆 proxy） ----------
    # 同时满足：靠近中心、低速、姿态稳定、双支撑脚接触，则给予小奖励
    dist_thresh = 0.5
    vel_thresh = 0.2
    angle_thresh = 0.1
    ang_vel_thresh = 0.1

    if (next_dist < dist_thresh and
        abs(vx) < vel_thresh and
        abs(vy) < vel_thresh and
        abs(angle) < angle_thresh and
        abs(ang_vel) < ang_vel_thresh and
        left_contact > 0.5 and right_contact > 0.5):
        soft_landing_proxy = 1.0
    else:
        soft_landing_proxy = 0.0

    # ---------- 总奖励 ----------
    w_progress = 5.0
    w_stab = 1.0      # stability_penalty 内部已含负号，直接加
    w_soft = 1.0

    total_reward = (
        w_progress * progress_delta +
        w_stab * stability_penalty +
        w_soft * soft_landing_proxy
    )

    components = {
        "progress_delta_reward": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-1771.110673, len=213.500000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| angular_vel_penalty | -0.001080 | 0.001080 | 0.999997 | -0.001080 |
| energy_penalty | -0.020271 | 0.020271 | 0.999996 | -0.020271 |
| potential_shaping | 0.130193 | 0.130394 | 0.999998 | 0.130193 |
| soft_landing_proxy | 0.000705 | 0.000705 | 0.001510 | 0.000705 |
| total_reward | 0.110252 | 0.113534 | 1.000000 | 0.110252 |
| generated_reward | 0.110252 | 0.113534 | 1.000000 | 0.110252 |
| original_env_reward | -4.382563 | 4.911105 | 1.000000 | -4.382563 |

## Distribution
- score: mean=-1771.110673, min=-5945.060097, max=-574.418700
- episode_length: mean=213.500000
- early_terminal (<150 steps + score<-50): 6/10 (60%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -107.10 | -107.10 | 0.00 | 74.20 | progress_delta_reward=0.016 soft_landing_proxy=0.005 stability_penalty=-0.142 | new_best |
| 2 | progress_delta_reward + soft_landing_proxy + stability_penalty | -111.45 | -107.10 | -4.35 | 74.10 | progress_delta_reward=0.016 soft_landing_proxy=0.003 stability_penalty=-0.058 | no_meaningful_improvement |
| 3 | angular_vel_penalty + energy_penalty + potential_shaping + soft_landing_proxy | -1771.11 | -107.10 | -1664.01 | 213.50 | angular_vel_penalty=-0.001 energy_penalty=-0.020 potential_shaping=0.130 soft_landing_proxy=0.001 | no_meaningful_improvement |
| 4 | angular_vel_penalty + energy_penalty + potential_shaping + soft_landing_proxy | 179.44 | 242.09 | -62.64 | 812.60 | angular_vel_penalty=-0.001 energy_penalty=-0.007 potential_shaping=0.045 soft_landing_proxy=0.336 | stop_after_solved_drop_keep_best |

```
