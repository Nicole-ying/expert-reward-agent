# Response Record

# 诊断报告

## evidence
- 历史得分：iter1=278.88（forward=1.0）→ iter2=285.19（forward=1.5，**历史最佳**）→ iter3=281.28 final（forward=2.0）→ iter4=29.83（死区铰链崩塌）→ iter5=180.28（tilt⁴ 尾项）。
- iter5 的失败机制明确：在 iter2 平滑二次平衡项上加了 `4.0*tilt**4`，checkpoint 趋势显示 `balance_penalty` 在 cp50/cp60 飙升到 -189.7/-206.3，`forward_reward` 在 cp80 掉到 209，说明 agent 为躲避四次尾项而减速、走路变保守，final forward ep_sum 从 iter2 的 756 掉到 606（平均速度约 0.53 m/s），score 掉到 180。结论：**tilt 惩罚只能维持 iter2 的二次基座，不能再加重**。
- iter3 的 checkpoint 趋势给出另一条关键信息：forward=2.0 时 cp70~cp80 的 checkpoint score 达到 299.7/302.6，但 final policy 掉到 281.28，且 `score_range=[-51.748, 300.544]` 出现摔倒回合 → **2.0 太快，最终策略脆弱、跨种子不稳定**。
- 由此推断最优速度/稳定性权衡点在 1.5（稳，285）与 2.0（快但脆，final 281）之间。当前 best=285.19，距目标 300 差 14.8 分（≈4.9%），属于「接近目标」区间。

## behavior_diagnosis
- iter5 策略被 tilt⁴ 尾项过度约束，学到「降低速度以保平衡」的保守步态，平均速度 ~0.53 m/s，score 180，是不健康的行为。
- iter2 策略是健康基线：稳定前进 ~0.68 m/s，balance_penalty magnitude_share 仅 4.1%，没有摔倒灾难样本。
- 需要的是在 iter2 的稳定步态上略微提速，而不是改变平衡惩罚形态。

## signal_completeness
- 必要职责完备：`forward_progress`（线性前向速度）与 `balance_and_fall_prevention`（tilt² + angvel² + vertvel²）都在。
- 本轮不引入 energy/gait 等条件职责，避免在接近目标时增加新的、未经验证的信号干扰。

## selected_level
- **L1**。理由：best 距目标 <10%，此前两次 L2（iter4 死区铰链、iter5 四次尾项）均显著回退；当前唯一需要的是在已验证良好的速度/稳定性权衡点上做系数微调。

## selected_intervention
- 目标组件：`forward_reward`。
- 具体修改：`forward_reward = 1.75 * next_obs[2]`（历史值为 1.0 / 1.5 / 2.0，本轮取 1.5 与 2.0 之间的新值 1.75）。
- 同时**删除 iter5 的 `4.0*tilt**4` 尾项**，恢复 iter2 验证过的平滑二次平衡基座（`1.0*tilt² + 0.2*angvel² + 0.1*vertvel²`），不再额外惩罚危险区。这不是简单回到 iter2（iter2 forward=1.5），而是以 iter2 的平衡形态为基础、把速度权衡点推向略快。

## falsifiable_hypothesis
- 1.5 太慢（285.19）、2.0 太快且 final 策略脆弱（281.28 / cp100 298.8），取 1.75 应把平均速度推到约 0.74~0.76 m/s：比 iter2 快、又低于 iter3 的脆弱速度，从而在不显著增加摔倒的前提下提高 mean score，预计向 290~298 靠拢，且 final 策略不再像 iter3 那样退化。

## expected_next_round
- 策略保持稳定步态，平均速度略快于 iter2；`forward_reward` ep_sum 预计约 850~900，`balance_penalty` magnitude_share 保持个位数；`early_terminal` 仍为 0/20；score 预计高于 285.19，逼近但可能仍略低于 300。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v6 reward: L1 微调 forward 系数 1.5 -> 1.75，删除 iter5 的四次尾项。
    #
    # 证据：iter2（forward=1.5 + 1.0*tilt² + 0.2*angvel² + 0.1*vertvel²）是历史最佳 285.19。
    # iter3（forward=2.0）checkpoint cp70~80 达 299.7/302.6，但 final 掉到 281.28，
    # 且 score_range 出现摔倒回合 -> 2.0 太快、最终策略脆弱。
    # iter5 在 iter2 基座上加了 4.0*tilt**4，balance_penalty 在 cp50/cp60 飙到 -189/-206，
    # forward ep_sum 从 756 掉到 606，agent 减速避险，score 掉到 180。
    #
    # 本轮 L1（单组件）：仅把 forward 系数取 1.5 与 2.0 之间的新值 1.75，
    # 同时恢复 iter2 验证过的平滑二次平衡项（不再叠四次尾项），
    # 在不引入新平衡形态的前提下把速度/稳定性权衡点推向略快。
    #
    # info 为空，不使用；无显式 success/failure 标志，不依赖 terminal flag。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    forward_reward = 1.75 * next_obs[2]

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
