## 诊断报告

### evidence
- 全局记忆：iter1(forward=1.0)=278.88 → **iter2(forward=1.5)=285.19（历史最佳）** → iter3(forward=2.0)=281.28 → iter4(死区铰链)=29.83 → iter5(tilt⁴ 尾项)=180.28 → iter6(forward=1.75)=120.03。
- 本轮 iter6 详细反馈：`score=120.03`，`len=850.6`，`terminated=20/20`，`score_range=[-95.30, 299.02]`——双峰分布，有摔倒回合也有接近满分回合；`forward_reward` magnitude_share=**97.2%**，`balance_penalty` 仅 2.8%。
- iter6 checkpoint 趋势剧烈震荡：`-30.1 → -37.8 → 114.9 → 183.4 → -80.1 → -89.0 → -34.9 → -84.7 → 173.7 → 90.7`，说明训练在「快而摔」与「慢而稳」两个盆地间反复横跳。
- 平均速度换算：iter2 forward ep_sum=756/len 1110≈**0.68 m/s**（285 分、稳定）；iter3=967/1160≈**0.83 m/s**（cp70~80 达 299.7/302.6 但 final 脆、出现摔倒）；iter6=592/850≈**0.70 m/s**（却大量摔倒、双峰）。
- 已穷尽的轴：线性 forward 系数 1.0/1.5/1.75/2.0 都已试过，1.5 最稳但 285，2.0 更快但脆，1.75 震荡；balance 的 hinge/quartic 强化也都失败。被拒草稿就是 iter2 精确回退，属于重复。

### behavior_diagnosis
iter6 学到的不是单一健康步态，而是一个**双峰/脆弱策略**：部分种子高速前进接近满分（299），部分种子高速失衡摔倒（-95）。根因是 forward 信号是**无界线性速度** `1.75*next_obs[2]`，magnitude_share 97.2% 完全支配奖励——agent 被持续推向「更快」，而 balance 惩罚（二次、仅 2.8%）在临近摔倒的倾角区（tilt≈0.5~0.6）只有 0.25~0.36，远不足以对抗速度诱惑。训练因此快速摔/慢速稳之间震荡，final 掉到 120。

### signal_completeness
- 前进信号：有，但**无界**，缺少「速度上限/饱和」来抑制对极端速度的追逐。
- 平衡信号：有（iter2 验证过的平滑二次形态），形态正确，但被无界速度信号淹没。
- 缺失的关键语义：**对速度的边际收益设上限**。这是「无界→归一化有界」的典型证据模式。

### selected_level
**L2**。当前 best=285 虽接近目标，但本轮 current=120 且训练震荡、双峰摔倒，问题不在「力度」而在主信号形态（无界线性速度）。被拒草稿已证明纯 L1 回退 iter2 是重复；线性系数轴也已穷尽，因此改主信号的数学形态。

### selected_intervention
把主信号 `forward_reward` 从无界线性 `1.75 * next_obs[2]` 改为**有界饱和形态**：

```
forward_reward = 2.0 * tanh(next_obs[2] / 0.7)
```

- 在 0~0.7 m/s 区间保持强梯度（把 agent 从静止拉起来），健康巡航速度 0.68 m/s 附近每步约得 1.5。
- 超过 0.8~1.0 m/s 后边际收益快速衰减，上限 2.0，agent 不再有「越快越好」的无界激励。
- balance 保持 iter2 验证过的平滑二次形态不变（1.0·tilt² + 0.2·angvel² + 0.1·vertvel²），不重蹈 iter4 hinge / iter5 quartic 的覆辙。

### falsifiable_hypothesis
把速度边际收益封顶后，「再快一点」不再带来超额回报，agent 会放弃通过极端速度换取分数的路径，转而收敛到 iter2 那样的稳定中速步态（≈0.6~0.8 m/s、低倾角），从而消除 iter6 的震荡与摔倒双峰。预期 checkpoint 在 cp60 后单调上升、不再大幅回摆。

### expected_next_round
策略收敛为单一稳定巡航步态：平均前向速度 0.6~0.8 m/s，tilt 保持在安全区，摔倒回合显著减少，score 稳定回到 280~290 区间，且 checkpoint 趋势不再剧烈震荡。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v7 reward (L2): 主信号从「无界线性速度」改为「有界饱和速度」，根除高速追逐。
    #
    # 证据：iter6(forward=1.75 线性) magnitude_share=97.2% 支配奖励，checkpoint
    # 在快而摔/慢而稳两盆地间震荡（-30→-38→115→183→-80→-89→-35→-85→174→91），
    # final=120、score_range=[-95,299] 双峰，多数种子高速摔倒。
    # iter3(2.0 线性) cp70~80 达 299.7/302.6 但 final 脆；iter2(1.5 线性) 最稳 285。
    # 线性系数轴 1.0/1.5/1.75/2.0 已穷尽；本轮改主信号形态为 tanh 饱和，
    # 把「越快越好」改为「达到巡航速度后边际收益衰减」，上限约 2.0/step。
    #
    # balance 保持 iter2 验证过的平滑二次形态，不重蹈 iter4 死区铰链 / iter5 四次尾项覆辙。
    #
    # info 为空，不使用；无显式 success/failure 标志，不依赖 terminal flag。
    _ = training_progress

    # 主学习信号：有界前向速度奖励。
    # 健康巡航速度约 0.68 m/s（iter2 最佳 285 的平均速度）。
    # 用 tanh(v/0.7) 使 0~0.7 区间梯度强、超速后边际收益衰减，避免极端速度支配奖励。
    v = next_obs[2]
    v_norm = v / 0.7
    # clamp 归一化输入，避免极端速度导致 exp 溢出
    v_norm = max(-3.0, min(3.0, v_norm))
    e2 = 2.718281828 ** (2.0 * v_norm)
    tanh_v = (e2 - 1.0) / (e2 + 1.0)
    forward_reward = 2.0 * tanh_v

    # 平滑连续平衡正则（iter2 精确形态）：二次基座，全程有梯度，无死区、无悬崖。
    tilt_penalty = 1.0 * next_obs[0] ** 2
    angular_velocity_penalty = 0.2 * next_obs[1] ** 2
    vertical_velocity_penalty = 0.1 * next_obs[3] ** 2

    balance_penalty = (
        tilt_penalty
        + angular_velocity_penalty
        + vertical_velocity_penalty
    )

    total_reward = forward_reward - balance_penalty

    components = {
        "forward_reward": forward_reward,
        "balance_penalty": -balance_penalty,
    }

    return float(total_reward), components
```