# Response Record

## 诊断报告

### evidence
- 全局记忆：iter1(1.0)=278.88 → **iter2(1.5)=285.19（历史最佳）** → iter3(2.0)=281.28 → iter4(死区铰链)=29.83 → iter5(tilt⁴)=180.28 → iter6(线性1.75)=120.03 → **iter7(tanh 饱和)=274.39**。
- 本轮 iter7：score=274.39，len=1069.8，terminated=20/20，truncated=0/20，`early_terminal=0/20`，`score_range=[33.78, 305.58]`。
- 组件：`forward_reward` ep_sum=1197.46（signed_share 99.4%、magnitude_share 99.5%），`balance_penalty` ep_sum=-6.48（0.5%），两者 active_rate 均 100%。
- 由 forward ep_sum 反推巡航速度：1197.46/1069.8≈1.119/step → tanh(v/0.7)=0.5597 → v≈0.442 m/s。对比 iter2：756.2/1110.5/1.5≈0.454 m/s。**iter7 巡航速度略低于历史最佳 iter2**，而 len 也略短（1069.8 vs 1110.5）。
- checkpoint 趋势：score 全程震荡（-112.8→-102→-95.8→167.9→-15.3→-76.3→1.6→192.5→108→277），训练过程仍不稳定；但 final（cp100=277）与最终反馈（274）接近，最终策略已收敛到较稳的行走，不再出现 iter6 那种 -95 的灾难性摔倒。

### 上一轮改善归因（+154 分）
iter6→iter7 唯一语义改动是主信号从**无界线性 `1.75*v`** 改为**有界饱和 `2.0*tanh(v/0.7)`**。证据：iter6 `score_range=[-95.30, 299.02]`（双峰，多粒种子高速追速后摔倒），final=120；iter7 `score_range=[33.78, 305.58]`，最低分从 -95 提升到 +33.8，`early_terminal=0`。tanh 饱和把「越快奖励越多」改成「超过巡航速度后边际收益衰减」，根除了极端高速追逐，这是本轮改善的来源。

### behavior_diagnosis
当前策略已学到**稳定但略保守的行走**：平衡惩罚只有 -6.48（比 iter2 的 -32.2 小 5 倍），说明步态很平滑、躯干倾斜小；但巡航速度 0.442 m/s 低于 iter2 的 0.454 m/s，len 也略短。score_range 显示仍有至少一粒种子早期摔倒（33.78），拖低均值。整体是健康的行走策略，不健康点在于**前进驱动略偏弱**，导致「速度×存活步数」的总前进量不足以稳定站上 300。

### signal_completeness
必要职责基本完备：`forward_progress`（next_obs[2] 前向速度）和 `balance_and_fall_prevention`（hull_angle/angvel/vertvel 二次惩罚）都在。不缺主信号；当前只是 forward 信号的力度比 iter2 最佳态略弱，导致巡航速度偏低。不需要新增组件。

### selected_level
**L1**。理由：(1) 当前 274.39 距目标 300 差 25.6（8.5%<10%），处于「接近目标」区间，规则要求优先 L1 微调；(2) 上一轮 tanh 饱和形态已被验证有效（把 120 拉到 274），不应再动数学形态；(3) 历史教训：iter4 死区铰链、iter5 四次尾项都是在接近目标时做 L2 导致崩塌。

### selected_intervention
目标组件：`forward_reward`。仅把 tanh 饱和主信号的乘子从 `2.0` 微调到 `2.2`（+10% 前进驱动），平衡项保持 iter2/iter7 验证过的平滑二次形态不变。这等价于在不破坏饱和防爆机制的前提下，把「速度/稳定性」权衡点向更快一侧移动一小步。

### falsifiable_hypothesis
当前巡航速度 0.442 m/s 低于 iter2（0.454），而官方得分近似正比于「速度×存活步数」。把 forward 乘子提到 2.2 后，整个速度区间的前向边际收益提高约 10%，RL 会把巡航速度从 ~0.44 推向 ~0.47–0.48 m/s；由于 tanh 上限只从 2.0 提到 2.2（仍有界），不会重演 iter6 的无界高速追逐。预计总前进量上升、得分向 290–300 靠拢，同时 `early_terminal` 保持 0。

### expected_next_round
下一轮策略预期仍是稳定行走，但巡航速度提高、`forward_reward` ep_sum 约增加 ~10%，`balance_penalty` 因速度带来的自然倾斜而略有上升（仍远小于 iter5 的危险量级）；score 预计 285–295，`score_range` 最低分保持正数，不再出现灾难性摔倒。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v8 reward (L1): 在 iter7 验证过的 tanh 饱和主信号上，把前向驱动乘子 2.0 -> 2.2。
    #
    # 证据：iter7(tanh, 2.0) 把 iter6 的无界线性灾难(120, 双峰 -95..299)救回 274，
    # 巡航速度 ~0.442 m/s，略低于历史最佳 iter2(线性1.5) 的 ~0.454 m/s。
    # 官方得分近似正比于「速度 x 存活步数」，当前缺口主要在前进驱动略弱。
    # 本轮只做 L1 微调：+10% forward 力度，把速度/稳定性权衡点略推向更快，
    # 同时保留 tanh 饱和（上限 2.2，仍有界），避免重演 iter6 高速追逐摔倒。
    #
    # balance 保持 iter2/iter7 验证过的平滑二次形态，不重蹈 iter4 死区铰链 / iter5 四次尾项覆辙。
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
    forward_reward = 2.2 * tanh_v

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
