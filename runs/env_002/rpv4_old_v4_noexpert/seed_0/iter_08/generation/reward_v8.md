# 诊断报告

**evidence**
- 全局记忆：`iter_6` 为 best `299.43`（`len=1323`，`terminated=20/20`，`score_range=[296.91, 301.78]`），距目标 300 仅 **0.19%**。`iter_7` 仅把 forward amplitude `2.0→2.05`，得分跌到 `238.55`，`len` 缩到 `1189`，`score_range=[-45.2, 291.2]`。
- `iter_6` checkpoint 趋势在 cp70 后稳定在 251→305→305→303；`iter_7` 趋势出现 cp90 骤降到 26.1 的波动（与 iter_5 的失败模式一致）。
- 组件历史：`energy_penalty` 在 iter_6 为 `-29.36`，iter_7 为 `-35.57`（同样系数 0.02，但总惩罚变差）；按步长折算 `-0.0222/step → -0.0299/step`，说明 iter_7 策略使用了更大 torque。
- `forward_progress` 在 iter_6 为 `593.6`，iter_7 为 `556.8`；按步长折算 `0.449/step → 0.468/step`，说明 iter_7 每步更快但总路程更短——典型“proxy 提高但外部任务不升”。

**behavior_diagnosis**
- iter_7 的策略学会了**更激进但更短命**的步态：单步 forward reward 和 torque 都更高，但 episode 更早结束（`len 1323→1189`），最终外部得分方差变大、均值下跌。这是 forward amplitude 过强重新打开的高 torque/速度 farming，不健康。

**signal_completeness**
- 必要职责完备：`forward_progress`（主目标）、`torso_stability`（防摔）、`vertical_velocity_penalty`（起伏抑制）、`energy_penalty`（效率/平滑）都在。缺的不是新信号，而是 **forward 与正则项的相对平衡**：iter_7 把 forward 拉高后，能量正则的相对约束被稀释。

**selected_level**
- **L1**。best `299.43` 与目标差距 0.19%（<10%），历史也显示接近目标时做 L2 越改越差；本轮只调系数，不改数学形态。

**selected_intervention**
- 目标组件：`forward_progress` + `energy_penalty` 的平衡。
- 具体修改：
  1. 将 forward amplitude 从 iter_7 的 `2.05` 恢复到 iter_6 已验证的 `2.0`（保持 tanh(0.60·v) 形状不变）。
  2. 把 `energy_penalty` 系数 `-0.02 → -0.025`，强化 torque 正则，直接压制 iter_7 出现的高 torque 通道。
- 这是与 iter_6（energy=-0.02）和 iter_7（amplitude=2.05, energy=-0.02）都不同的新干预。

**falsifiable_hypothesis**
- iter_7 的失败来自 forward 相对权重过高 → torque 增大 → 步态更激进 → 更早摔倒。恢复 forward 到验证过的平衡点、同时略微提高能量惩罚，会把策略推离“高 torque 短命”局部最优，回到平滑可持续步态：预计 `energy_penalty/step` 下降、`len` 回升到 ~1300、`forward_progress` 总量回升到 ~590，低分 seed（296.9）补齐到 300，均值 ≥ 300。

**expected_next_round**
- 策略恢复 iter_6 式平滑高效步态，episode 长度回到 ~1300，终止仍为 20/20，`score_range` 重新收紧在 300 附近，均值突破 300。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # Forward: restore iter_6's proven amplitude 2.0 / slope 0.60.
    # iter_7's 2.05 amplitude re-opened high-torque speed farming (shorter
    # episodes, higher per-step energy, lower external score).
    if horizontal_velocity >= 0.0:
        v = min(horizontal_velocity, 4.0)
        # tanh(0.60 * v) = (e^(1.2*v) - 1) / (e^(1.2*v) + 1)
        e2v = 2.718281828 ** (1.2 * v)
        forward_progress = 2.0 * (e2v - 1.0) / (e2v + 1.0)
    else:
        # Asymmetric backward penalty: moving backward costs more than forward
        # gains, discouraging in-place oscillation.
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Stability and vertical terms held at iter_6's proven coefficients.
    torso_stability = -0.15 * (hull_angle ** 2) - 0.05 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    # L1 micro-tune: strengthen torque-energy regularization -0.02 -> -0.025.
    # iter_7's energy/step worsened (-0.0222 -> -0.0299), showing the policy
    # moved to larger torques under the stronger forward signal. A slightly
    # stronger energy penalty restores the forward-vs-efficiency balance
    # without changing any mathematical form.
    energy_penalty = -0.025 * (
        action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2
    )

    total_reward = (
        forward_progress
        + torso_stability
        + vertical_velocity_penalty
        + energy_penalty
    )

    components = {
        "forward_progress": forward_progress,
        "torso_stability": torso_stability,
        "vertical_velocity_penalty": vertical_velocity_penalty,
        "energy_penalty": energy_penalty,
    }

    return float(total_reward), components
```