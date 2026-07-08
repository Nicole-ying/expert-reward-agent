# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next positions
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    body_angle = obs[4]  # current body angle

    # Euclidean distances to the target platform
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # Main learning signal: progress toward the target
    progress = dist_curr - dist_next

    # Stability constraint: penalize large body tilt
    tilt_penalty = -0.1 * abs(body_angle)

    total_reward = progress + tilt_penalty

    components = {
        'distance_progress': progress,
        'tilt_penalty': tilt_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 组件与角色

| 组件名 | 类型 | 角色 | 说明 |
|--------|------|------|------|
| `distance_progress` | 主学习信号 | 告诉 agent “靠近目标平台” | 每一步奖励等于到平台中心欧氏距离的减小量。该信号每步都有梯度，直接驱动 agent 缩短与目标的距离，是实现“快速到达目标平台”的核心引导。 |
| `tilt_penalty` | 稳定约束 | 维持合理姿态，防止过度倾斜 | 用当前机体角度的绝对值进行轻量惩罚（权重 0.1）。aim：鼓励 agent 保持接近垂直的姿态，降低因倾斜过大撞击地面或超出视口的风险。权重明显小于主信号，避免过早扼杀探索中的必要姿态调整。 |

## 未使用 terminal_success_reward / terminal_failure_penalty 的原因

- 环境 **不存在显式的 success 或 failure 标志**（`explicit_success_flag_available: false`），且 `info` 字段为空。
- 不能从观测中直接推断“机体 no awake”等终止条件，因此无法可靠构造终端奖励，强行使用会导致奖励不可解释或误导。

## 未使用的组件及留待后续迭代

- **动作/引擎代价**：环境鼓励节约主引擎推力，但按照原则 5（v1 先学完成任务，再优化效率），暂时不加入任何动作惩罚，以免 agent 为避免惩罚而拒绝点火，导致探索不足。
- **接触奖励与软成功代理**：双腿接触 (`left_support_contact`, `right_support_contact`) 是任务完成的重要线索，但单纯奖励二值接触容易被反复轻触利用。v1 阶段不引入，后续可通过与低速、近中心、小姿态等多条件组合，构造安全的 soft proxy。
- **着陆速度约束**：同样留到 v2 或 v3，与接触信号配合，实现“轻柔触地”的鼓励。

## 训练后应观察的 failure mode

1. **高速撞击地面**：agent 学会快速下降但没有学会减速，导致临近平台时速度过大而 crash。需后续增加着陆速度惩罚或触地前减速引导。
2. **在目标附近震荡/徘徊**：仅有距离进度可能让 agent 接近平台时反复来回移动以持续获取较小的进度奖励，而不是稳定着陆。可能需要加入稳定性要求。
3. **过度依赖主引擎**：即使没有引擎代价，agent 可能持续点火保持姿态或快速下降，浪费燃料。后续引入动作代价可缓解。
4. **倾斜失败**：`tilt_penalty` 权重较小，agent 可能在降落过程中产生较大倾角并触发 `crash_or_body_contact`。若此现象频繁，可在后续增大约束权重或与高度关联。
