## 1. 诊断报告

### evidence（关键数据与历史）

- 历史分数轨迹：`iter_1=-17.27`（无门控，向前摔倒）→ `iter_2=262.56`（加门控 0.5）→ `iter_3=255.93`（门控收紧 0.4，惩罚加强，负向）→ `iter_4=287.49`（门控放宽 0.6，+31.6）→ `iter_5=291.96`（门控放宽 0.7，+4.5）。
- 对比 `read_checkpoint_trend(4)` 与 `read_checkpoint_trend(5)`：门控 0.6→0.7 的改善不只是最终分数，更在于训练过程显著变稳：
  - iter_4：cp10%=-103.5、cp50%=-107.8、cp70%=-95.7、cp90%=38.7，训练极不稳定，最后才跳升到 289.5。
  - iter_5：cp10%=-48.8、cp40%=91.9、cp60%=263.2、cp70%=289.6、cp100%=293.3，路径平滑得多，且最终分数更高。
- 值得注意的是 iter_5 的 `forward_velocity_reward` ep_sum 最终为 809.4，反而低于 iter_4 的 818.9，而外部分数更高。说明放宽门控让策略选择了更直立、更健康的步态，proxy 前进奖励略降，但外部任务分上升——当前 0.7 门控对自然前倾仍有轻微过压。
- iter_5 仍存在一次 mid-training 崩盘：cp40% 91.9 → cp50% -51.9（`upright_stability_penalty` 同步 spike 到 -36.6），之后恢复。说明门控宽度仍有放宽空间来进一步平滑探索。
- 当前 score=291.96，目标 300，差距 8.04（2.7%），`score_range=[284.4, 293.4]`，`terminated=20/20`、无 truncation、无 reward_errors。
- 组件份额：`forward_velocity_reward` 占 99.8% magnitude_share（active 99.2%）；`upright_stability_penalty` 仅 -0.2%（active 26.5%）；`vertical_oscillation_penalty` 全程为 0（死组件，但无害）。

### behavior_diagnosis

策略已学会稳定双足前进，不再是 iter_1 的"向前摔倒刷速度"。20/20 均在环境终止条件下结束（摔倒或到地形末端），无早摔（early_terminal 0/20），分数区间窄且接近目标，说明策略接近收敛。当前主要瓶颈不是缺少信号，而是门控对自然前倾仍略偏紧：外部分数在 forward reward 略降时反而上升，说明"更直立/更自然"的步态比"更高速但更前倾"更贴合外部任务评分。继续小幅放宽门控是本轮最安全的推进方向。

### signal_completeness

必要职责完备：`forward_velocity_reward`（主前进信号）+ `upright_stability_penalty`（防摔护栏）+ `vertical_oscillation_penalty`（死但无害）。当前没有显式 fall/success flag 与真实能耗信号，环境卡也禁止依赖 info。能量/对称性等次目标在接近目标阶段不宜新增组件，避免引入新的 proxy 冲突。

### selected_level

**L1**。差距 2.7%（<10%），且历史证据明确指向同一数学形态下的一个系数——直立门控宽度是过去两轮唯一且连续有效的杠杆，本轮只需在该系数上继续微调，不改任何数学表达式。

### selected_intervention

目标组件：`forward_velocity_reward` 中的 `upright_gate`。把门控宽度 `0.7 → 0.8`，其余全部保持不变（速度温度 1.5、前向 scale 1.5、稳定性惩罚 -4.0/-0.4、垂直死区 0.6 均不动）。

### falsifiable_hypothesis

0.7 的门控仍略微过度压制自然前倾：策略为了保持高 gate 值而选择偏直立、偏慢的步态。放宽到 0.8 后，中等前倾区间获得的 forward reward 更高，训练早期探索更平滑（预计 cp50 附近的崩盘减轻或消失），策略能学到稍微更快、仍不触发摔倒护栏的步态，从而把外部分数从 292 推向 300。若下一轮 `forward_velocity_reward` 显著上升而外部分数下降，说明又回到了"前倾刷速度"，假设被证伪。

### expected_next_round

预计 score 提升到 ~295–298；训练 checkpoint 曲线更平滑（中段负向尖峰减弱）；`forward_velocity_reward` ep_sum 维持或略降（更健康的步态）；`terminated` 仍 20/20；`upright_stability_penalty` 仍为小幅度护栏，不出现大幅 spike。

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：稳定前进。
    # 水平速度只有在躯干保持直立时才被全额奖励，防止"向前摔倒"被误奖为前进。
    vx = float(next_obs[2])
    speed_reward = 1.0 - 2.718281828 ** (-max(vx, 0.0) * 1.5)

    # 直立门控：历史证据显示放宽宽度（0.4 -> 0.6 -> 0.7）持续改善外部分数，
    # 且 iter_5 外部分数上升的同时 forward reward 略降，说明门控仍轻微过压自然前倾。
    # 本轮继续沿同一方向 L1 微调（0.7 -> 0.8），让中等前倾获得更自然的前进奖励，
    # 同时保留强力的兜底惩罚，防止回到"向前摔倒刷前进奖励"。
    hull_angle = float(next_obs[0])
    upright_gate = 2.718281828 ** (-(hull_angle / 0.8) ** 2)

    forward_velocity_reward = 1.5 * speed_reward * upright_gate

    # 稳定性约束：保持上一轮加强后的力度（角度 -4.0，角速度 -0.4），
    # 作为放宽门控后的兜底护栏，防止向前摔倒。
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