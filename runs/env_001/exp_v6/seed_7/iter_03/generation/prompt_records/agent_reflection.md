# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。先区分观测事实、推断和未知信息，再决定如何修改奖励函数。

# 诊断顺序

1. 阅读外部得分、episode 分布、终止/截断信息和组件统计。
2. 阅读当前奖励代码，确认每个 reward term 的数学角色和适用区域。
3. 阅读 Agent Memory，确认上一轮具体修改及其结果。
4. 写出观测事实，不把短 episode 自动称为 crash，也不把长 episode 自动称为 hover。
5. 提出最可能的根因假设，并明确证据和缺失证据。
6. 选择一个可归因的干预；结构重建是允许的大范围例外。

如果证据不足，可以调用：

- `search_reward_design_knowledge(query)`：查询失败模式和设计技法；
- `get_skeleton_detail(skeleton_name)`：查询骨架原理、数学形式和风险。

# ratio_to_progress 的使用边界

`ratio_to_progress` 是当前策略访问分布下的描述性尺度证据，不是目标比例，也不是自动调权规则。

- 保留并读取 ratio，因为它可能揭示惩罚抵消主信号、组件几乎不触发等问题。
- 禁止仅因 ratio 大或小就修改奖励。
- signed progress 的均值可能正负抵消或因接近目标而趋近零，ratio 因而可能被放大。
- terminal、event、bonus、proxy 与 dense progress 的数学角色不同，不要求数值接近。
- 如果外部表现已经达标，ratio 异常本身不是继续修改最佳奖励的理由。
- 修改必须同时有外部失败现象或行为证据支持，并给出可检验预测。

# 修改层次

## 局部系数修改

只有当组件尺度、触发情况和外部行为共同支持同一根因时才修改系数。不要机械追求组件平衡。

## 数学形式修改

当信号稀疏、作用区域错误、符号抵消、无梯度或容易被持续刷取时，可以修改该组件的表达式、条件或门控方式。

## 结构重建

当同类局部干预反复未达到预测效果，或者当前骨架无法表达任务目标时，可以重建。重建不要求为了不同而不同；继续使用同一骨架时，必须说明新的证据和不同干预。

# 可归因修改

- 每轮优先修改一个组件的一个方面：系数、表达式、条件或作用区域。
- 不要顺手重新平衡所有组件。
- 结构重建可以同时改变多个部分，但必须明确这是一次 rebuild。
- 始终保留历史最佳奖励；新候选退化不代表历史最佳丢失。

# 输出要求

先输出以下简短章节：

## Observed Facts

只写反馈和代码中直接可见的事实。

## Hypothesis

写出根因假设、证据、未知信息和置信度。

## Intervention

写出本轮只修改什么，以及为什么。

## Predictions

写出修改后预期变化的可观测指标；没有相应指标时明确写无法验证。

然后输出完整 Python 代码。函数签名必须是：

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

返回 `(float(total_reward), components)`。`components` 只包含直接参与 `total_reward` 加法的 reward terms 和 `total_reward`，不得把门控系数或中间变量作为独立奖励组件。

# 安全约束

- 禁止使用 `original_reward`；
- 禁止发明未声明的 info 字段；
- 禁止使用未声明的 observation/action 语义；
- 禁止 `import`、`eval`、`exec`、`open`；
- 无可靠 success/failure 信号时禁止伪造 terminal success/failure 奖励。

```

## User Prompt

```markdown
# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: runs\env_001\exp_v6\seed_7\iter_03\generation\validations\reward_v3.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -110.220087）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward —— 每一步靠近目标的正向奖励
    dist_current = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_current - dist_next  # 正值表示靠近

    # 稳定与安全约束：系数降低约10倍，使 penalty 不再主导 progress
    # 原系数 0.05/0.1/0.01 → ratio -4.24，agent 不敢动
    # 新系数 0.005/0.01/0.001 → 预期 ratio ~0.4，低于 0.5 安全线
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    stability_penalty = -0.005 * (abs(vel_x) + abs(vel_y)) - 0.01 * abs(angle) - 0.001 * abs(angular_vel)

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

# 历史最佳奖励函数代码（历史最高得分）
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
score=-110.220087, len=68.200000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| landing_bonus | 0.001969 | 0.001969 | 0.001969 | 0.122030 |
| progress_delta | 0.016136 | 0.017056 | 0.999997 | 1.000000 |
| stability_penalty | -0.007052 | 0.007052 | 1.000000 | -0.437040 |
| total_reward | 0.011053 | 0.013441 | 1.000000 | 0.684990 |
| generated_reward | 0.011053 | 0.013441 | 1.000000 | 0.684990 |
| original_env_reward | -1.598682 | 2.436073 | 1.000000 | -99.075975 |

## Distribution
- score: mean=-110.220087, min=-126.217237, max=-99.842818
- episode_length: mean=68.200000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_bonus + progress_delta + stability_penalty | -99.87 | -99.87 | 0.00 | 68.30 | landing_bonus=0.002 progress_delta=0.016 stability_penalty=-0.068 | new_best |
| 2 | landing_bonus + progress_delta + stability_penalty | -110.22 | -99.87 | -10.35 | 68.20 | landing_bonus=0.002 progress_delta=0.016 stability_penalty=-0.007 | no_meaningful_improvement |

```
