# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 1. 提取观察量 ──
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    vel_x, vel_y = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 2. 主学习信号：进度差奖励 ──
    dist_old = (x ** 2 + y ** 2) ** 0.5
    dist_new = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_old - dist_new

    # ── 3. 轻量稳定约束 ──
    stability_penalty = -0.01 * (abs(vel_x) + abs(vel_y)) \
                        -0.01 * abs(angle) \
                        -0.01 * abs(angular_vel)

    # ── 4. 软着陆近似信号 ──
    # 当飞行器满足“靠近目标、低速、姿态竖直、双腿接触”时给予小额奖励
    near_target = dist_new < 0.15
    low_speed = (vel_x ** 2 + vel_y ** 2) ** 0.5 < 0.2
    upright = abs(angle) < 0.1
    both_legs_down = (left_contact > 0.5) and (right_contact > 0.5)

    soft_landing_proxy = 0.5 if (near_target and low_speed and upright and both_legs_down) else 0.0

    # ── 组合总奖励 ──
    total_reward = progress + stability_penalty + soft_landing_proxy

    # ── 组件字典 ──
    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件与角色

| 组件 | 角色 | 说明 |
|------|------|------|
| `progress` | **主学习信号** | 每一步越接近目标（欧氏距离减少）获得正奖励，反之受惩罚。直接驱动飞行器向目标移动。 |
| `stability_penalty` | **稳定/安全约束** | 对平移速度、机体角度与角速度施加小权重惩罚，抑制高速冲撞、剧烈翻滚，引导平稳飞行。 |
| `soft_landing_proxy` | **任务完成近似信号** | 当飞行器满足“靠近目标、低速、竖直、双腿接触”时给予小额奖励，鼓励最终稳定着陆，但不伪造成功标志。 |

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境卡片明确 `explicit_success_flag_available: false`，且 `info` 字典为空，无可信的终点成功/失败标志。直接使用这些组件会诱导模型依赖不可靠的信号，因此 v1 完全禁用，仅依靠过程奖励与近似信号。

## 哪些组件留到后续迭代
- **终端成功/失败奖励**：待环境 wrapper 明确暴露 `success` 标记后加入。
- **能量/动作惩罚**：v1 不加入，避免 agent 畏惧动作。后续加入 `energy_penalty = -0.001 * (action != 0)`。
- **时间效率惩罚**：若训练后 agent 在目标附近徘徊，可加入 `time_penalty = -0.001`。
- **距离锚定奖励**：如果仅使用 `progress_delta` 导致在目标附近震荡，可加入小权重 `distance_reward = -0.1 * dist_new` 作为 anchor。
- **门控奖励**：若安全约束被主信号淹没，可考虑通过速度/姿态阈值设计门控，但 v1 避免过度复杂。

## 训练后应观察的 failure mode
1. **goal_near_oscillation**：飞行器在目标附近来回摆动，progress 信号接近零但仍不落地。此时应检查 `soft_landing_proxy` 触发频率，可能需要收紧条件或增大该奖励。
2. **high_reward_without_success**：总奖励很高但实际未稳定着陆（例如恰好停在垫子附近但未接触）。应检查 `soft_landing_proxy` 是否被 hack，后续可强化接触与位置联合判断。
3. **fast_crash_near_goal**：飞行器高速冲向垫子，虽然距离快速缩短，但导致碰撞或弹起。此时需增大 `stability_penalty` 权重。
4. **afraid_to_drop**：稳定惩罚过强导致飞行器悬停在高处不敢下降。需降低 penalty 系数或添加基于高度的势能引导（后续迭代）。