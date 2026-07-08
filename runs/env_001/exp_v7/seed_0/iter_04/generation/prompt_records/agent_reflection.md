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

**层次 1：改系数。** 通过 abs_contrib_% 判断组件间相对量级。惩罚项 abs_contrib_% 显著偏高且外部得分差 → 考虑削弱。bonus 类 abs_contrib_% 天然偏高，只要 nonzero_rate 正常且得分不差就不用修。nonzero_rate < 2% → 增大权重或放宽条件。

**层次 2：改数学形式。** 同一个系数反复调还是不行，说明当前数学形式本身有问题。考虑改变信号的计算方式——但每次只改一个组件的形式，下一轮看效果。

**层次 3：换骨架。** 以下情况停止在层次 1/2 上打转，直接换主信号框架：
- 同一骨架家族已迭代 2 轮以上，且最佳得分仍未超过 target 的 25%。
- 或者已经改过数学形式（层次 2）但得分没有实质性改善。

# 奖励函数迭代的通用原则

以下原则来自大量实验，与环境无关。

## 原则 1：相对贡献是通用语言

不关心组件系数的绝对值。关心组件之间的相对大小：
- 查看 feedback 的 `abs_contrib_%` 列：`100 * 该组件 abs_mean / 所有组件 abs_mean 之和`。所有组件加起来 = 100%。这个指标衡量每个组件在总奖励量级中的占比——不关心系数绝对值，关心组件间的**相对体量**。
- 学习信号应该是最大的正向贡献者。
- 约束/惩罚应该是弱背景信号——如果它的 abs_contrib_% 显著超过学习信号，agent 可能选择"不动"来避免惩罚，而非"行动"来获取奖励。
- 终端/事件型奖励（偶尔触发）的 abs_contrib_% 天然偏大，这不代表它有问题。只要触发率正常、外部得分不差，高比率不是 bug。

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
错误信息：Reward v4 failed validation: runs\env_001\exp_v7\seed_0\iter_04\generation\validations\reward_v4.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -1368.988936）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 势能塑形 (Potential-Based Shaping) ===
    # 原理：F = γ * Φ(s') - Φ(s)，是唯一保证最优策略不变的塑形方式。
    # Φ(s) = -(dist + α*speed + β*|angle|)，同时引导靠近、减速、姿态稳定。
    # 相比 distance_reward = -dist'（只关心绝对位置），势能塑形奖励"改善量"，
    # 提供更稠密的梯度：靠近有奖、减速有奖、摆正有奖。

    # 当前状态
    dist = ((obs[0] - 0.0) ** 2 + (obs[1] - 0.0) ** 2) ** 0.5
    speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    angle = abs(obs[4])

    # 下一状态
    dist_next = ((next_obs[0] - 0.0) ** 2 + (next_obs[1] - 0.0) ** 2) ** 0.5
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_next = abs(next_obs[4])

    # 势能函数：负的加权状态代价
    alpha_speed = 0.5   # 速度权重，使 speed 分量与 dist 分量量级可比
    beta_angle = 0.5    # 角度权重，使 angle 分量与 dist 分量量级可比
    phi_now = -(dist + alpha_speed * speed + beta_angle * angle)
    phi_next = -(dist_next + alpha_speed * speed_next + beta_angle * angle_next)

    gamma = 0.99  # 接近 1，轻微折扣提供时间偏好
    potential_shaping = gamma * phi_next - phi_now

    # === 独立接触奖励（加法，不被乘积归零） ===
    # 左右腿接触均为 [0,1] 连续值
    contact_bonus = (next_obs[6] + next_obs[7]) * 0.1

    # === 总奖励 ===
    total_reward = potential_shaping + contact_bonus

    components = {
        'potential_shaping': potential_shaping,
        'contact_bonus': contact_bonus,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置为原点 (0,0)
    # 计算到目标的欧氏距离（使用 next_obs）
    dist_to_target = ((next_obs[0] - 0.0) ** 2 + (next_obs[1] - 0.0) ** 2) ** 0.5
    
    # 主学习信号：负距离，引导飞行器持续靠近目标
    distance_reward = -dist_to_target

    # 稳定性惩罚：轻量抑制高速、大角度和角速度，促进安全减速和姿态稳定
    speed_norm = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angvel_abs = abs(next_obs[5])
    
    lambda_vel = 0.02
    lambda_angle = 0.02
    lambda_av   = 0.02
    
    stability_penalty = -(lambda_vel * speed_norm + lambda_angle * angle_abs + lambda_av * angvel_abs)

    # 软着陆近似奖励：多条件组合，只有当飞行器非常接近目标、速度极低、姿态稳定且双支撑脚均接触时才触发
    threshold_dist   = 0.3
    threshold_speed  = 0.1
    threshold_angle  = 0.05
    contact_left     = next_obs[6]
    contact_right    = next_obs[7]
    
    if (dist_to_target < threshold_dist and 
        speed_norm < threshold_speed and 
        angle_abs < threshold_angle and 
        contact_left == 1.0 and 
        contact_right == 1.0):
        soft_landing_proxy = 10.0
    else:
        soft_landing_proxy = 0.0

    # 总奖励
    total_reward = distance_reward + stability_penalty + soft_landing_proxy

    # 记录各组件
    components = {
        'distance_reward': distance_reward,
        'stability_penalty': stability_penalty,
        'soft_landing_proxy': soft_landing_proxy,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-1368.988936, len=177.600000, errors=0

## Component evidence

Column definitions:
- `mean`: per-step average of the component value (signed). Positive=reward, negative=penalty.
- `abs_mean`: per-step average of the absolute value. Measures magnitude regardless of sign.
- `nonzero_rate`: fraction of steps where the component value is non-zero (|value| > 1e-12).
- `abs_contrib_%`: this component's share of total reward magnitude. Computed as `100 * abs_mean_of_this_component / sum(abs_mean_of_all_components)`. All components sum to 100%. Use this to judge relative scale — a penalty with high abs_contrib_% may be dominating the learning signal.

| component | mean | abs_mean | nonzero_rate | abs_contrib_% |
|-----------|------|----------|-------------|--------------|
| contact_bonus | 0.000750 | 0.000750 | 0.004858 | 0.011621 |
| potential_shaping | 0.126059 | 0.126599 | 1.000000 | 1.962533 |
| total_reward | 0.126809 | 0.127322 | 1.000000 | 1.973750 |
| generated_reward | 0.126809 | 0.127322 | 1.000000 | 1.973750 |
| original_env_reward | -5.606683 | 6.068778 | 1.000000 | 94.078345 |

## Distribution
- score: mean=-1368.988936, min=-8273.598986, max=-356.722928
- episode_length: mean=177.600000
- early_terminal (<150 steps + score<-50): 15/20 (75%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_reward + soft_landing_proxy + stability_penalty | -111.87 | -111.87 | 0.00 | 68.45 | distance_reward=-0.971 soft_landing_proxy=0.028 stability_penalty=-0.026 | new_best |
| 2 | distance_reward + soft_landing_continuous + stability_penalty | -115.29 | -111.87 | -3.42 | 68.40 | distance_reward=-0.969 soft_landing_continuous=0.013 stability_penalty=-0.027 | no_meaningful_improvement |
| 3 | contact_bonus + potential_shaping | -1368.99 | -111.87 | -1257.12 | 177.60 | contact_bonus=0.001 potential_shaping=0.126 | no_meaningful_improvement |

```
