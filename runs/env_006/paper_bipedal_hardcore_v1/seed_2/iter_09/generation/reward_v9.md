# 设计理由

**信号覆盖审计**：当前奖励函数使用了 hull_angle、hull_angular_velocity（在 health gate 中）和 horizontal_speed，但缺失对姿态倾斜的直接惩罚。leg contact、vertical_speed 等观测未被使用，但历史 iter4 移除双脚离地惩罚后 len 回升到 377，说明直接使用接触信号可能风险较高。更紧迫的是：历史 iter7（hinge_penalty + forward + energy）达到了 best score -52.19 和 len 401.70，而 iter8 去除了 hinge penalty 后 len 暴跌至 217。这证明 **当前缺少对倾角的明确惩罚是存活下降的核心原因**。

**行为诊断**：agent 在 level 地面能较快前进（progress 组件贡献大），但 health gate 只是按比例衰减进度奖励，无法为“接近摔倒”的状态提供足够的负梯度。因此 agent 常常倾斜过大而摔倒，导致存活长度下降、总 return 偏低。

**修改内容**：添加 **hinge balance penalty** 组件。采用线性超出式惩罚：`-0.5 * max(0, |hull_angle| - 0.4)`。该惩罚只在倾角超过 0.4 rad（约 23°）时激活，给 agent 一个明确的安全区，同时不会在正常小摆动时施加不必要压力。系数选择 0.5，使惩罚 per-step 控制在主信号（约 0.275/步）的 0.3x 以下（实际超出 0.1 rad 时惩罚 0.05/步，0.2 rad 时 0.1/步），不会压制前进奖励。

**与已有 gate 的关系**：gate 在倾角 0.4 rad 时已经衰减 progress 到约 0.38x，叠加 hinge 惩罚形成双重抑制，会更强地推动 agent 返回安全姿态。这比单纯 gate 更为有效，又比全尺度二次惩罚更温和。

**为什么不改动 progress 或 gate**：如果移除 gate 而只靠 hinge，将回到 iter6 水平（len 尚可但 score 不如 iter7）。保留 gate 与 hinge 共存是一种稳妥的增量修改，符合正常模式只改一个组件的约束。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract useful signals ----------
    horizontal_speed = obs[2]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]

    # ---------- health gate: close to 1 when upright, decays when tilting ----------
    denom = 1.0 + 10.0 * hull_angle * hull_angle + 0.1 * hull_angular_velocity * hull_angular_velocity
    health_gate = 1.0 / denom

    # ---------- forward progress ----------
    fwd_speed = max(0.0, horizontal_speed)
    progress_component = 1.0 * fwd_speed * health_gate

    # ---------- action regularisation ----------
    action_sum_sq = action[0]*action[0] + action[1]*action[1] + action[2]*action[2] + action[3]*action[3]
    action_penalty = -0.01 * action_sum_sq

    # ---------- hinge balance penalty: explicit tilt cost beyond safe zone ----------
    tilt_magnitude = abs(hull_angle)
    safe_threshold = 0.4   # ~23 degrees
    excess_tilt = max(0.0, tilt_magnitude - safe_threshold)
    hinge_balance_penalty = -0.5 * excess_tilt

    # ---------- total reward ----------
    total_reward = progress_component + action_penalty + hinge_balance_penalty

    components = {
        "progress": progress_component,
        "action_penalty": action_penalty,
        "hinge_balance_penalty": hinge_balance_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺失——缺少对姿态倾斜的直接惩罚，导致存活长度从 401 骤降至 217。
- **behavior**: agent 在平坦段较快前进，但极易因倾斜过大摔倒，存活不稳定。
- **signal**: 缺少对 hull_angle 的明确负梯度（health gate 衰减不足以防止极限倾斜）。
- **level**: Level 2（增加 hinge balance penalty 组件）。
- **hypothesis**: 显式的倾角惩罚将填补安全梯度缺口，降低摔倒率，恢复存活长度并提升总得分（参考 iter7 有效模式）。
- **risk**: 与 health gate 叠加可能使 agent 过度保守，但通过适中阈值（0.4 rad）和低系数（0.5）可避免抑制正常摆动。