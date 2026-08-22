## 诊断报告

### evidence
- 历史得分：iter1=278.88 → iter2=285.19（forward=1.5，历史最佳）→ iter3=281.28（forward=2.0 回退）→ iter4=29.83（死区铰链崩塌）。目标 300，最佳差距 4.9%，属于「接近目标」区间。
- iter2 组件：`forward_reward` magnitude_share=95.9%，`balance_penalty`=4.1%；balance 是平滑二次形式（`1.0*tilt² + 0.2*angvel² + 0.1*vertvel²`），全程有梯度，工作良好。
- iter3 关键证据（历史反思已记录）：`score_range=[-51.748, 300.544]`，非摔倒回合均值≈298.8，缺口主要来自**后期摔倒**，而不是速度。这也是 iter4 尝试修摔倒的原因。
- iter4 崩坏机理：`fall_risk_penalty = 40*max(0, |tilt|-0.4)²` 是死区+悬崖。checkpoint 显示 cp10 该惩罚冲到 -451（早期探索被重罚），cp20~40 agent 学会站立不动（forward≈2~7）来逃避越界；最终 `fall_risk_penalty` ep_sum=-158、magnitude_share=35.3%、active_rate 仅 7.4%——惩罚由少数大倾角样本支配，训练出的策略在固定评估种子上脆弱（训练 cp100 forward=544，评估 forward 仅 288，terminated=10/20）。
- `get_component_history` 确认：`balance_penalty` 在 iter1~3 量级为 -12~-32（温和、健康），iter4 换成 hinge 后 `fall_risk_penalty` 骤然到 -158（被 exploit/支配）。

### behavior_diagnosis
iter4 策略是不健康的：死区铰链在 0.4 rad 以下没有梯度，agent 没有学会「保持躯干安全余量」，而是学会了「要么站着不动逃避惩罚，要么贴着 0.4 阈值快速走但缺乏鲁棒性」。最终表现为 terminated=10/20、score=29.83、评估种子泛化差。相比之下 iter2 的平滑二次正则让 agent 稳定前进、balance 占比仅 4.1%，是历史验证过的健康结构。

### signal_completeness
必要职责基本完备：`forward_progress`（前向速度）已有，`balance_and_fall_prevention` 已有。缺的不是新信号，而是**摔倒防护的数学形态错了**：iter4 用死区悬崖，丢掉了 0.4 以下的连续梯度；而 iter2 的纯二次在危险区（|tilt|>0.5）惩罚又太弱，没能抑制后期摔倒。

### selected_level
**L2**（单组件改数学形态）。理由：直接回退 iter2 会被判为语义重复，纯 L1 调系数同样是 iter2 的语义同构；但最佳距目标仅 4.9%，所以只做**保守的单组件形态修改**，保留已验证的 forward=1.5 和二次基座，不动主信号框架。

### selected_intervention
- 保留：`forward_reward = 1.5 * next_obs[2]`（iter2 最佳系数）、`0.2*angvel² + 0.1*vertvel²`。
- 删除：iter4 的死区铰链 `40*max(0, |tilt|-0.4)²`。
- 修改 `balance_penalty` 的 tilt 项形态：由纯二次 `1.0*tilt²` 改为 **二次基座 + 四次尾项**：
  `tilt_penalty = 1.0 * tilt² + 4.0 * tilt⁴`。
  - 二次基座保留 iter2 全程连续梯度（正常步态约 0.17 rad 不受影响）。
  - 四次尾项在危险区（|tilt|>0.5）快速增强但平滑无死区、无悬崖（对比 iter4 的 40 系数铰链温和得多），给「临近摔倒」提供更强的连续梯度。

### falsifiable_hypothesis
四次尾项全程有梯度（无死区），因此不会复现 iter4「站立不动逃避悬崖」的策略；它在 0.5~0.8 rad 危险区把惩罚推到与 forward 相当或更强的量级，教会 agent 主动保持躯干安全余量、减少后期摔倒。**验证标准**：下一轮 score 回升到 275 以上、`balance_penalty` magnitude_share 回到 <10%（不再被支配）、forward 占比回到 ~90%+；若 balance 惩罚再次爆炸或 agent 又站立不动，则说明 4.0 尾项系数过强，应降为 2.0。

### expected_next_round
预计策略恢复为 iter2 式稳定前进，且比 iter2 更少在后期出现大倾角摔倒：score 预计回到 275~295 区间，terminated 的失败比例下降，`forward_reward` 重新成为主导信号，`balance_penalty` 温和增长但远低于 iter4 的 35%。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v5 reward: 删除 iter4 的死区铰链悬崖，恢复平滑连续平衡正则，
    # 并给 tilt 惩罚增加四次尾项，强化「临近摔倒」区域的连续梯度。
    #
    # 证据：iter4 用 40*max(0,|tilt|-0.4)^2 死区铰链后 score 285.19 -> 29.83。
    # checkpoint cp10 fall_risk=-451 重罚早期探索，cp20~40 agent 站立不动逃避惩罚，
    # 最终 fall_risk_penalty magnitude_share=35.3% 支配奖励，策略在评估种子脆弱。
    # 根因：死区在 0.4 以下无梯度，且 40 系数形成悬崖。
    #
    # 本轮 L2（单组件）：保留 iter2 验证过的 forward=1.5 与二次基座，
    # 把 tilt 惩罚改为「二次基座 + 四次尾项」：正常步态(~0.17 rad)几乎不变，
    # 危险区(|tilt|>0.5)平滑增强、无死区，教会 agent 保持躯干安全余量。
    #
    # info 为空，不使用；无显式 success/failure 标志，不依赖 terminal flag。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    # 保留已验证的最优速度系数 1.5（iter2 最佳 285.19）。
    forward_reward = 1.5 * next_obs[2]

    # 平滑连续平衡正则：二次基座 + 四次尾项。
    tilt = next_obs[0]
    tilt_base = 1.0 * tilt ** 2          # iter2 验证过的全程连续梯度
    tilt_tail = 4.0 * tilt ** 4          # 危险区快速增强，无死区、无悬崖

    angular_velocity_penalty = 0.2 * next_obs[1] ** 2
    vertical_velocity_penalty = 0.1 * next_obs[3] ** 2

    balance_penalty = (
        tilt_base
        + tilt_tail
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