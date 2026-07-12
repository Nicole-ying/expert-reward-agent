# Response Record

`evidence`: 当前迭代使用`max(0, next_settled - current_settled)`配合系数400.0，settlement_bonus的episode_sum_mean高达16726（magnitude_share 99.9%），但外部score仅89，存在严重proxy脱节；best迭代使用persistent state-value形态（2.0 * settled_state），score达197.6且组件占比正常；`contact_reward_hacking`知识确认高接触奖励+低外部得分=漏洞信号。

`behavior_diagnosis`: 策略在transition-based settlement_bonus的`max(0, delta)`形态和系数400下找到漏洞：通过在settled_score上制造小幅正delta（如速度/姿态轻微振荡后恢复）反复领取奖励，内部奖励爆炸但未转化为真实着陆完成，13/20 episode因超时截断而非成功终止。

`signal_completeness`: 必要职责（goal_proximity、settling、orientation）信号均存在且可达，但settlement_bonus的数学形态（max(0, delta)）破坏了可证伪性——允许单边积累正奖励而无需真正完成着陆。

`selected_level`: Level 2——证据直接否定当前`max(0, delta)`形态（创建了可被振荡利用的非对称反馈），需做`state_to_improvement`结构变换，将单边max修正为完整势能差分（无max），消除重复领取漏洞。

`selected_intervention`: 唯一目标组件为settlement_bonus，从`400.0 * max(0, next_settled - current_settled)`变换为`150.0 * (next_settled - current_settled)`——完整势能差分，正负对称，单episode总贡献严格有界（≤150.0），杜绝振荡套利。

`falsifiable_hypothesis`: 完整势能差分消除了单边正delta积累漏洞，settlement_bonus的episode_sum_mean应从16726骤降至合理范围（≤150），同时正负对称的反馈迫使策略必须真正完成着陆才能获得净正奖励；proximity_reward在接近阶段提供连续引导，settlement_bonus在接触瞬间提供强梯度，二者协同时不应出现proxy徘徊。

`expected_next_round`: settlement_bonus的episode_sum_mean应大幅下降（预期0-150范围），signed_share从99.9%显著降低，proximity_reward和speed_penalty_gated相对占比上升；外部score应恢复到best水平（~197）或更高；terminated比例应上升（更多成功着陆而非超时截断）。

`main_risk`: 完整势能差分在接近阶段会产生小幅负奖励（settled_score因速度波动而下降时），噪声增加可能使早期探索变慢；若150.0系数在接触事件上产生的单步奖励过大（如delta=0.8时+120），可能导致值函数估计不稳定。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft_contact, nright_contact = next_obs

    # ----------------------------------------------------------------
    # 1. Goal proximity: potential-based delta shaping
    #    Positive when moving toward target, negative when moving away
    # ----------------------------------------------------------------
    distance = (x**2 + y**2)**0.5
    next_distance = (nx**2 + ny**2)**0.5
    proximity_reward = 2.0 * (distance - next_distance)

    # ----------------------------------------------------------------
    # 2. Orientation penalty: keep body upright and stable
    # ----------------------------------------------------------------
    orientation_penalty = -0.1 * (angle**2) - 0.1 * (angvel**2)

    # ----------------------------------------------------------------
    # 3. Soft landing: punish high velocities when near the pad
    # ----------------------------------------------------------------
    proximity_gate = 1.0 / (1.0 + 5.0 * distance)
    speed_sq = vx**2 + vy**2
    speed_penalty_gated = -0.5 * speed_sq * proximity_gate

    # ----------------------------------------------------------------
    # 4. Settlement improvement: proper potential-based delta
    #    Uses full delta (NOT max(0, delta)) to prevent exploitation
    #    by oscillation. Total per episode is strictly bounded:
    #    150.0 * (final_settled - initial_settled) <= 150.0
    #    Positive when becoming more settled, negative when regressing,
    #    forcing the agent to truly complete the landing for net gain.
    # ----------------------------------------------------------------
    # Current settled score (from obs)
    current_avg_contact = (left_contact + right_contact) / 2.0
    current_prox_gate = 1.0 / (1.0 + 5.0 * distance)
    current_speed_sq = vx**2 + vy**2
    current_stillness = 1.0 / (1.0 + 10.0 * (current_speed_sq + angvel**2))
    current_settled = current_avg_contact * current_prox_gate * current_stillness

    # Next settled score (from next_obs)
    next_avg_contact = (nleft_contact + nright_contact) / 2.0
    next_prox_gate = 1.0 / (1.0 + 5.0 * next_distance)
    next_speed_sq = nvx**2 + nvy**2
    next_stillness = 1.0 / (1.0 + 10.0 * (next_speed_sq + nangvel**2))
    next_settled = next_avg_contact * next_prox_gate * next_stillness

    # Full potential-based delta: symmetric, no exploitation by oscillation
    settlement_bonus = 150.0 * (next_settled - current_settled)

    # ----------------------------------------------------------------
    # Combine
    # ----------------------------------------------------------------
    total_reward = (
        proximity_reward +
        orientation_penalty +
        speed_penalty_gated +
        settlement_bonus
    )

    components = {
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "speed_penalty_gated": speed_penalty_gated,
        "settlement_bonus": settlement_bonus
    }

    return float(total_reward), components
```
