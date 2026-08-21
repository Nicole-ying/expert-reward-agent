## 1. 诊断报告

### evidence（关键数据与历史）

- **iter_1 → iter_2 的改善来源明确**：对比 `read_reward_code(1)` 与 `read_reward_code(2)`，上一轮唯一实质修改是在 `forward_velocity_reward` 上乘了一个直立门控 `upright_gate = exp(-(hull_angle/0.5)^2)`。
- 该修改的证据链：
  - iter_1 最终策略 `score=-17.27`，`len=375.45`，`terminated=20/20`，且 `early_terminal 14/20`；同时 `forward_velocity_reward` 高达 `ep_sum=222.6`（97.2% share）。说明上一轮策略学会了“向前摔倒”：水平速度很大、拿到大量前进奖励，但外部得分极差、episode 很早结束。
  - iter_2 加入门控后，`score=262.56`，`len=1342.3`，`forward_velocity_reward ep_sum=814.5`，`upright_stability_penalty ep_sum=-0.264`、`active_rate=4.5%`。策略从“向前摔倒”变成“直立行走”，所以 **+279 分的改善来自直立门控**。
- 当前最终评估分布：`score_range=[-6.27, 279.34]`，`terminated=20/20`。best 279.34 已经非常接近目标 300（差距 6.9% < 10%），但均值 262.56 被极低分 episode（-6.27）拖低。
- 训练过程趋势（`read_checkpoint_trend(2)`）显示脆弱性：总体 score 在 cp80%=251 → cp90%=-104 → cp100%=270 之间剧烈震荡；cp90% 时 `forward_velocity_reward` 骤降到 2.26、`upright_stability_penalty` 活跃率冲到 83.3%（ep_sum=-4.25），说明训练中反复掉入“摔倒吸引子”。

### behavior_diagnosis

- 当前策略在多数种子下已经学会“直立前进”：`upright_stability_penalty` 活跃率只有 4.5%、ep_sum 接近 0，说明绝大多数时间躯干角度很小，属于健康的直立步态。
- 但策略**不稳健**：最终评估里仍有极低分 episode（-6.27），训练曲线在 cp90% 出现摔倒崩塌。best 接近 300 而均值只有 262.56，说明瓶颈不是“走不到 279”，而是**偶发摔倒把均值拉低**。
- `forward_velocity_reward` 占了 100% magnitude share，upright 惩罚几乎为 0。稳定性实际主要靠 forward reward 里的门控在承担，而独立的 `upright_stability_penalty` 太弱（ep_sum=-0.264），无法在接近摔倒时提供足够的负反馈把策略拉回。

### signal_completeness

- `forward_velocity_reward`：完备且有效（有门控的主前进信号）。✓
- `upright_stability_penalty`（防摔/稳定性职责）：存在但**力度不足**。在接近摔倒的中等倾斜区间（0.3–0.5 rad），当前惩罚量级（约 -0.2~-0.5）远小于前进奖励量级（约 0.6–1.4），不足以形成强排斥。
- `vertical_oscillation_penalty`：全程 0，是死组件，但无害。
- 能量/动作平滑：缺失，属于次目标，当前不应引入以免干扰前进主目标。

### selected_level

**L1**。理由：
1. best 得分 279.34 距离目标 300 仅 6.9%（< 10%），按规则优先 L1 微调，不做数学形态改动。
2. 改善来源已能明确回答（iter_2 的直立门控），但当前不需要换框架，只需要在已被验证有效的稳定性方向上加强力度。
3. 症状是“偶发摔倒拖低均值”，属于力度问题（惩罚太弱、门控稍宽），不是方向或公式形态错误。

### selected_intervention

单一语义：**加强防摔/稳定性约束**（不改变任何数学表达式，只调参数）。

- `upright_gate` 宽度 `0.5 → 0.4`：让中等倾斜（0.3–0.5 rad）时前进奖励下降更快，更早抑制“为了速度向前倾”的趋势。
- `upright_stability_penalty` 力度增强：
  - 角度惩罚系数 `-2.0 → -4.0`
  - 角速度惩罚系数 `-0.15 → -0.4`

这两个子调整服务于同一个语义：让策略更强烈地避开中等倾斜和接近摔倒状态，减少偶发摔倒。

### falsifiable_hypothesis

- 当前策略在好种子里已能走到 ~279，但中等倾斜区间（0.3–0.5 rad）仍被奖励容忍，导致部分种子滑入摔倒吸引子。
- 把门控收紧 + 把稳定性惩罚翻倍，会显著降低中等倾斜和近摔倒状态的价值/增大其代价，使策略在训练和评估中更一致地留在直立吸引子内。
- 预期效果：极低分 episode 消失或减少，均值向 best（~279）靠拢，训练曲线在 cp90% 附近不再出现 -100 级的崩塌；前进奖励 magnitude 可能小幅下降（步态更稳但略慢），但外部得分均值应上升。

### expected_next_round

- 最终策略保持直立行走的比例更高，`upright_stability_penalty` 的活跃率在好 episode 中仍保持低位，但评估 score_range 的下界会明显抬升（不再出现 -6 这类早摔）。
- 均值 score 预计从 262.56 向 275–290 区间抬升；best 可能维持在 ~279 附近或略微提升。
- 若下一轮均值已逼近 best 但 best 不再上升，说明瓶颈切换到“速度/步幅”，届时再考虑在前进速度温度或系数上做 L1 微调，而非继续收紧稳定性。

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：稳定前进。
    # 水平速度只有在躯干保持直立时才被全额奖励，防止"向前摔倒"被误奖为前进。
    vx = float(next_obs[2])
    speed_reward = 1.0 - 2.718281828 ** (-max(vx, 0.0) * 1.5)

    # 直立门控：本轮收紧宽度（0.5 -> 0.4），让中等倾斜更早失去前进奖励。
    hull_angle = float(next_obs[0])
    upright_gate = 2.718281828 ** (-(hull_angle / 0.4) ** 2)

    forward_velocity_reward = 1.5 * speed_reward * upright_gate

    # 稳定性约束：加强力度（角度 -2.0 -> -4.0，角速度 -0.15 -> -0.4），
    # 使接近摔倒的倾斜/旋转得到更强的负反馈。
    angle_margin = max(0.0, abs(hull_angle) - 0.15)
    angular_vel_margin = max(0.0, abs(float(next_obs[1])) - 1.0)
    upright_stability_penalty = (
        -4.0 * angle_margin ** 2
        - 0.4 * angular_vel_margin ** 2
    )

    # 上下弹跳约束：保持不动。
    vertical_margin = max(0.0, abs(float(next_obs[3])) - 0.6)
    vertical_oscillation_penalty = -0.3 * vertical_margin ** 2

    total_reward = (
        forward_velocity_reward
        + upright_stability_penalty
        + vertical_oscillation_penalty
    )

    components = {
        "forward_velocity_reward": float(forward_velocity_reward),
        "upright_stability_penalty": float(upright_stability_penalty),
        "vertical_oscillation_penalty": float(vertical_oscillation_penalty),
    }

    return float(total_reward), components
```