# Response Record

## 1. 诊断报告

### evidence（关键数据与历史）

- 历史分数轨迹（memory）：`iter_5=291.96`（best，纯高斯门控宽 0.7）→ `iter_6=216.50`（门控放宽 0.8，-75.46）→ `iter_7=253.09`（平台 0.25 + 高斯尾 0.45，+36.59）。
- **本轮改善 >30 分的归因（必须回答）**：`iter_6 → iter_7` 唯一实质修改是把纯高斯 `width=0.8` 换成“平台 + 高斯尾”。证据链：`iter_6` 的门控在 0.7 rad 处 gate=0.465、0.8 rad 处 gate=0.368，太松，导致 cp50~cp80 连续四段全负、最终 216.5；`iter_7` 用 tail sigma 0.45 恢复了 0.7 rad 处 gate=0.368（与 best 的 iter_5 完全一致），重新压住了大角度“向前摔倒刷速度”，因此 score 回升到 253.1，cp90 一度到 275.9。
- 但 `iter_7` 仍远低于 best 292：`read_checkpoint_trend(7)` 显示 cp50=-99.2（深度塌缩），同时 `upright_stability_penalty` 在 cp50 的 active_rate=78.6%、ep_sum=-104.2；对比 best 的 `iter_5` cp50=-51.9、active=25.0%、ep_sum=-36.6。说明 iter_7 的门控在**中等前倾区间（0.25~0.7 rad）仍比 iter_5 松**，训练中期依然诱使策略用中等前倾刷前进奖励，随后被罚崩。
- 数值对比（中等角度 gate）：0.4 rad 处 iter_5=0.721、iter_7=0.895；0.5 rad 处 iter_5=0.600、iter_7=0.734；0.7 rad 处两者都=0.368。iter_7 在 0.25~0.7 区间系统性更松，这是它与 best 的唯一实质性差距。
- `vertical_oscillation_penalty` 全程 0（死组件），但无害，不是本轮瓶颈。

### behavior_diagnosis

最终策略能走且不倒（terminated=20/20，len≈1184），但最终策略学到的是“带中等前倾的快速步态”：它靠 `forward_velocity_reward`（ep_sum=802，share 99.8%）驱动，`upright_stability_penalty` 只有 -1.58（active 3.5%）。内部奖励构成与 best iter_5 接近（iter5 forward=809、penalty=-1.43），但外部分数 253 vs 292，说明中等前倾步态让它在内部奖励上几乎无损，却在外部的“走得远/稳”上明显变差。训练过程不健康：cp50 深跌到 -99 说明“中等前倾刷速度”仍是一个训练中途被反复试出来、再被压下去的错误局部最优。

### signal_completeness

必要职责齐全：前进主信号（`forward_velocity_reward`）、躯干稳定性护栏（`upright_stability_penalty`）、弹跳抑制（死但无害）都在。不缺信号，缺的是**门控在中等角度区间的松紧度校准**。

### selected_level

**L1**。理由：best=291.96，距目标 300 差距 2.7% < 10%，按规则必须优先系数微调；且历史证据已把问题精确定位到“中等角度门控过松”，无需改数学形态或换框架，只需在两个系数上回调。

### selected_intervention

目标组件：`upright_gate`（直立门控）。具体修改：平台宽度 `0.25 → 0.15`，尾部 sigma `0.45 → 0.55`。

效果：0.7 rad 处 gate 保持 0.368（=iter_5），0.8 rad 处 gate=0.247（≈iter_5 的 0.271），但 0.4 rad 从 0.895 收紧到 0.813、0.5 rad 从 0.734 收紧到 0.667，把中等前倾区间拉回 iter_5 的轮廓，同时保留 ≤0.15 rad 的自然前倾全额奖励。

### falsifiable_hypothesis

如果 iter_7 相对 best 的退化来自“中等前倾区间被过度奖励导致训练中期 exploit + 最终步态偏倾”，那么把 0.25~0.7 rad 区间的 gate 收紧回 iter_5 轮廓后，cp50 的深跌应收窄（向 iter_5 的 -51.9 靠拢），最终策略应回到更直立的快速步态，外部分数预计回到 ~290 并有望冲击 300。若分数仍上不去，则说明问题不在中等角度松紧，而在于“平台+高斯尾”本身的形状（如平台边界曲率），下一轮再考虑 L2。

### expected_next_round

训练中期塌缩明显变浅，cp50 不再深跌；最终策略更接近 iter_5 的直立快走，`upright_stability_penalty` active_rate 回到 ~20-30% 的迭代中间水平、终态降到低位；score 预计 290 左右，best 有望被刷新。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：稳定前进。
    # 水平速度只有在躯干不接近摔倒时才被全额奖励，防止"向前摔倒"被误奖为前进。
    vx = float(next_obs[2])
    speed_reward = 1.0 - 2.718281828 ** (-max(vx, 0.0) * 1.5)

    # 直立门控：保留"平台 + 高斯尾"结构，但把中等前倾区间（0.25~0.7 rad）收紧回
    # iter_5 已验证的纯高斯 0.7 轮廓。
    # 证据：iter_5（纯高斯 0.7）=291.96 best；iter_6（纯高斯 0.8）=216.5 塌缩；
    # iter_7（平台 0.25 + tail 0.45）=253.1，中等角度仍比 iter_5 松，cp50 深跌到 -99。
    # 本轮 L1：平台 0.25->0.15，sigma 0.45->0.55。
    # 效果：0.7 rad 处 gate=0.368（与 iter_5 一致）、0.8 rad 处 gate=0.247（≈iter_5 0.271），
    # 但 0.4 rad 0.895->0.813、0.5 rad 0.734->0.667，把中等前倾的过度奖励拉回 iter_5 水平。
    hull_angle = float(next_obs[0])
    lean_margin = max(0.0, abs(hull_angle) - 0.15)
    upright_gate = 2.718281828 ** (-(lean_margin / 0.55) ** 2)

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
