## 1. 诊断报告

### evidence（关键数据与历史）

- 记忆表：`iter_5=291.96`（best，门控宽 0.7）→ `iter_6=216.50`（-75.46，唯一改动是门控宽 0.7→0.8）。目标 300，当前 best 差距 2.7%。
- `read_checkpoint_trend(5)` vs `read_checkpoint_trend(6)` 是核心证据：
  - iter_5：cp50 短暂回落 -51.9 后，cp60 即恢复到 263.2，cp70~cp100 稳定在 ~290。
  - iter_6：cp50~cp80 连续四段全负（-84.9 / -87.7 / -49.8 / -59.0），直到 cp90 才恢复，最终仅 252.8。
  - 同一时段 iter_6 的 `upright_stability_penalty` active_rate 冲到 28.2%/17.3%/25.2%/18.9%，说明训练中期策略反复进入过度前倾状态；而 iter_5 同期只有 25.0%/5.0%/5.6%/10.2%。
- 最终策略对比：iter_5 `upright_stability_penalty` active=26.5%/ep_sum=-1.43（频繁但轻微触线）；iter_6 active=2.8%/ep_sum=-4.97（罕见但剧烈触线）——iter_6 末期姿态更极端、更脆。
- `get_component_history('forward_velocity_reward')`：该组件在全部迭代中 magnitude_share 99%+，是唯一有效信号；`vertical_oscillation_penalty` 全程 0（阈值 0.6 从未触发，死组件）。

### behavior_diagnosis

iter_6（门控 0.8）的策略是**不健康的脆弱步态**：最终集里大部分时间躯干直立，但会罕见地剧烈偏离（active 2.8% 却贡献 -4.97），且训练中期出现长时间「前倾—摔倒—奖励塌缩」循环。episode length 从 iter_5 的 1217.6 降到 1132.25，20/20 终止，说明更早摔倒、前进距离更短。根本问题不是惩罚太弱，而是**高斯门控对任何前倾都乘性打折**——自然前倾（0.1~0.25 rad）也被衰减，逼迫策略在「速度」和「姿态」之间做不稳定的权衡；放宽到 0.8 后门控尾部又过松，导致大角度摔倒抑制失效。

### signal_completeness

必需职责基本齐备：前进（`next_obs[2]`）、姿态稳定（`next_obs[0]/[1]`）、垂直弹跳（`next_obs[3]`）。但存在两点缺陷：
1. 垂直弹跳惩罚是死组件（阈值 0.6 永不触发）；
2. 门控与稳定性惩罚**职责重叠**，门控以乘法方式重复惩罚姿态，是训练塌缩的放大器。

（能量惩罚属于外部分数维度，但当前阻塞点不是能耗，而是门控塌缩，本轮不动。）

### selected_level

**L2**——只改 `upright_gate` 一个组件的数学形态。

选择理由：
- 门控宽度这一 L1 轴已被穷尽：0.4→255.9、0.5→262.6、0.6→287.5、0.7→292.0、0.8→216.5，峰值在 0.7，任何「再调一个宽度」都落在已知区间内；直接回退 0.7 是重复（已被拒）。
- iter_6 的失败证明「纯高斯门控」本身有结构缺陷：它无法同时满足「自然前倾全额奖励」与「大角度强抑制」两个需求——这是宽度参数无法解决的形态问题。
- 本轮改动保守：**完整保留 iter_5 已验证的大角度抑制强度**（0.7 rad 处 gate 仍为 0.368），只把「自然前倾区间」改为平台，属于定向形态修正而非重写。

### selected_intervention

目标组件：`forward_velocity_reward` 中的 `upright_gate`。

具体修改：
- 原：`gate = exp(-(hull_angle / 0.7) ** 2)`（纯高斯，任何前倾都打折）。
- 新：`gate = exp(-(max(0, |hull_angle| - 0.25) / 0.45) ** 2)`（平台 + 高斯尾部）。

效果：`|angle| <= 0.25` 全额奖励自然前倾；`0.7 rad` 处 gate=0.368 与 iter_5 完全一致；`0.8+` 处抑制比 iter_5 更强（0.8 处 0.224 vs 0.271）。其余组件（速度奖励、稳定性惩罚、垂直惩罚）全部保持 iter_5 原样。

### falsifiable_hypothesis

如果 iter_6 的塌缩根因是「高斯门控衰减自然前倾 → 策略在速度与姿态间不稳定权衡」，那么平台门控在自然前倾区间移除该衰减、同时保留 iter_5 的大角度抑制，应使训练像 iter_5 一样在 cp60 前后稳定恢复、且最终分数 ≥ 291.96。**可证伪**：若平台过宽诱导策略在 0.25 rad 边缘堆前倾，`upright_stability_penalty` 的 active_rate 会在训练中期重新冲高、score 跌破 iter_5，则说明平台阈值应回调。

### expected_next_round

策略预计以更自然的躯干前倾换取更高平均前进速度，摔倒更晚、episode length 回升；checkpoint 趋势应在 cp60 前后稳定进入高分区，避免 iter_6 那种 cp50~cp80 的长时间负分塌缩；最终 score 应落在 291.96 附近或更高。

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：稳定前进。
    # 水平速度只有在躯干不接近摔倒时才被全额奖励，防止"向前摔倒"被误奖为前进。
    vx = float(next_obs[2])
    speed_reward = 1.0 - 2.718281828 ** (-max(vx, 0.0) * 1.5)

    # 直立门控：从纯高斯改为"平台 + 高斯尾部"。
    # 证据：门控宽 0.4->0.6->0.7 持续改善（255.9->287.5->292.0），
    # 但 0.8 触发训练中期塌缩（cp50~cp80 全负）并跌至 216.5。
    # 根因：纯高斯对任何前倾（包括 0.1~0.25 rad 的自然前倾）都乘性打折，
    # 逼迫策略在速度与姿态间不稳定权衡；单纯放宽宽度又削弱了摔倒抑制。
    # 本改法：自然前倾区间（<=0.25 rad）全额奖励，大角度保持 iter_5 已实证的抑制强度
    # （0.7 rad 处 gate=0.368），从而把"前进速度"与"躯干倾斜"两个职责解耦。
    hull_angle = float(next_obs[0])
    lean_margin = max(0.0, abs(hull_angle) - 0.25)
    upright_gate = 2.718281828 ** (-(lean_margin / 0.45) ** 2)

    forward_velocity_reward = 1.5 * speed_reward * upright_gate

    # 稳定性约束：保持 iter_5 的力度（角度 -4.0，角速度 -0.4），
    # 作为平台门控放宽自然前倾后的加法式兜底护栏，防止向前摔倒。
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