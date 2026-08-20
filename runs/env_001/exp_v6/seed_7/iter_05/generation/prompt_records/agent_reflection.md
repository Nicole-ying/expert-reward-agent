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

## Prediction Review

如果输入提供了“上一轮生成该奖励时的诊断、干预和预测”，逐项根据当前反馈标记 `supported`、`refuted` 或 `inconclusive`。如果没有上一轮预测，写 `not_available`。不得把未验证预测当成事实。

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
错误信息：Reward v5 failed validation: runs\env_001\exp_v6\seed_7\iter_05\generation\validations\reward_v5.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -109.076793）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # next_obs 解包
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    # x_vel = next_obs[2]   # 未直接使用，但保留以备扩展
    # y_vel = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. 距离奖励：基于到目标平台中心的欧氏距离，密集引导接近目标
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -distance

    # 2. 姿态惩罚：避免倾斜过大，鼓励竖直姿态
    angle_penalty = -abs(angle)

    # 3. 角速度惩罚：抑制快速旋转，提高稳定性
    angular_vel_penalty = -abs(ang_vel)

    # 4. 能量惩罚：减少引擎使用，满足节省燃料要求
    energy_penalty = 0.0
    if action == 2:               # 主引擎
        energy_penalty = -0.1
    elif action in (1, 3):        # 转向引擎
        energy_penalty = -0.05
    # action 0 不额外惩罚

    # 5. 时间惩罚：鼓励尽快完成任务
    time_penalty = -0.01

    # 总奖励为各项直接相加
    total_reward = (
        distance_reward
        + angle_penalty
        + angular_vel_penalty
        + energy_penalty
        + time_penalty
    )

    components = {
        "distance_reward": distance_reward,
        "angle_penalty": angle_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "energy_penalty": energy_penalty,
        "time_penalty": time_penalty,
        "total_reward": total_reward,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-109.076793, len=68.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| angle_penalty | -0.028193 | 0.028193 | 1.000000 | -0.028193 |
| angular_vel_penalty | -0.066917 | 0.066917 | 0.999934 | -0.066917 |
| distance_reward | -0.972708 | 0.972708 | 1.000000 | -0.972708 |
| energy_penalty | -0.010048 | 0.010048 | 0.188596 | -0.010048 |
| time_penalty | -0.010000 | 0.010000 | 1.000000 | -0.010000 |
| total_reward | -1.087866 | 1.087866 | 1.000000 | -1.087866 |
| generated_reward | -1.087866 | 1.087866 | 1.000000 | -1.087866 |
| original_env_reward | -1.577995 | 2.327043 | 1.000000 | -1.577995 |

## Distribution
- score: mean=-109.076793, min=-123.234285, max=-95.059093
- episode_length: mean=68.300000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 上一轮生成该奖励时的诊断、干预和预测
请先用当前训练反馈审查这些预测，输出 supported/refuted/inconclusive 及证据。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # next_obs 解包
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    # x_vel = next_obs[2]   # 未直接使用，但保留以备扩展
    # y_vel = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. 距离奖励：基于到目标平台中心的欧氏距离，密集引导接近目标
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -distance

    # 2. 姿态惩罚：避免倾斜过大，鼓励竖直姿态
    angle_penalty = -abs(angle)

    # 3. 角速度惩罚：抑制快速旋转，提高稳定性
    angular_vel_penalty = -abs(ang_vel)

    # 4. 能量惩罚：减少引擎使用，满足节省燃料要求
    energy_penalty = 0.0
    if action == 2:               # 主引擎
        energy_penalty = -0.1
    elif action in (1, 3):        # 转向引擎
        energy_penalty = -0.05
    # action 0 不额外惩罚

    # 5. 时间惩罚：鼓励尽快完成任务
    time_penalty = -0.01

    # 总奖励为各项直接相加
    total_reward = (
        distance_reward
        + angle_penalty
        + angular_vel_penalty
        + energy_penalty
        + time_penalty
    )

    components = {
        "distance_reward": distance_reward,
        "angle_penalty": angle_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "energy_penalty": energy_penalty,
        "time_penalty": time_penalty,
        "total_reward": total_reward,
    }

    return float(total_reward), components
```

---

### 设计说明

**主信号选择：distance_reward（绝对值距离惩罚）**  
前序迭代使用的 `progress_delta + landing_bonus + stability_penalty` 已失败，因此本次回避所有增量进步类骨架（包括 `progress_delta_reward` 和 `potential_based_shaping`），改用**静态距离惩罚**作为核心过程引导。每步给予 `-distance`，直接鼓励飞行器始终保持在目标平台附近，避免在目标上方悬停而不下降，同时也对远离目标的行为施加持续性负反馈。

**其他组件角色**  
- **angle_penalty & angular_vel_penalty**：利用 `body_angle` 和 `angular_velocity` 实现姿态稳定，防止翻滚或剧烈旋转，提升最终降落的平稳性。  
- **energy_penalty**：根据动作类型对使用主引擎和转向引擎分别施加小惩罚，响应“尽可能减少引擎推力”的任务要求。  
- **time_penalty**：每步固定负值，促使智能体以最短时间完成到达与稳定，同时抑制无意义的原地逗留。

**未使用的骨架及理由**  
- `terminal_success_reward` / `terminal_failure_penalty`：环境未提供任何显式的 `success` 或 `failure` 标志，信息接口不足以可靠实现。  
- `distance_reward` 的绝对值形式天然隐含了对越界（水平坐标过大）的惩罚，因此无需单独设计 `failure_penalty` 或 `gated_reward`。  
- `landing_bonus` 已在之前试验中被规避，且缺乏可靠的成功检测手段，故本次不予采用。  
- `potential_based_shaping` 虽为推荐候选，但其效果与 `progress_delta` 本质上同类，基于增量变化，为避免重复失败模式而推迟。

**预期需要观察的失败模式**  
- 智能体可能满足于在目标正上方小距离悬停，而迟迟不完成最终着陆（y≈0 但 x≈0 处 hover）。此时 `energy_penalty` 和 `time_penalty` 会持续施压，需观察其能否推动最终接触。  
- 如果着陆后 `body_not_awake_or_settled` 触发较慢，`time_penalty` 会持续累积，可能导致智能体在着陆瞬间仍受负奖励，但整体影响应有限。  
- 距离惩罚可能使智能体在学习早期过分保守，不敢探索更远区域；可通过调整相对权重缓解，但当前最小设计保留单一尺度以观察效果。  

**留待后续迭代的目标**  
- 对最终稳定着陆的显式肯定（如利用双腿接触+低速+小角度的组合推断成功），待观察到 `distance_reward` 主信号能产生可靠趋近行为后，再考虑通过辅助奖励注入，以避免当前强行添加不可靠的软成功奖励。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_bonus + progress_delta + stability_penalty | -99.87 | -99.87 | 0.00 | 68.30 | landing_bonus=0.002 progress_delta=0.016 stability_penalty=-0.068 | new_best |
| 2 | landing_bonus + progress_delta + stability_penalty | -110.22 | -99.87 | -10.35 | 68.20 | landing_bonus=0.002 progress_delta=0.016 stability_penalty=-0.007 | no_meaningful_improvement |
| 4 | angle_penalty + angular_vel_penalty + distance_reward + energy_penalty + time_penalty | -109.08 | -109.08 | 0.00 | 68.30 | angle_penalty=-0.028 angular_vel_penalty=-0.067 distance_reward=-0.973 energy_penalty=-0.010 time_penalty=-0.010 | new_best |

```
