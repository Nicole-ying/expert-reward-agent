`evidence`：20/20 terminated（无truncated），13/20为快速失败（<150步且score<-50）；proximity_reward以86% magnitude_share主导，均值-68.65；但mean distance≈0.95说明agent已接近目标区域，却在着陆阶段失败；contact_bonus active_rate仅1%、signed_share仅0.2%，着陆引导几乎不可达；上一轮为首次尝试，无历史干预冲突。

`behavior_diagnosis`：agent学会了向目标区域靠近（平均距离约0.95），但在接近后无法完成减速与稳定着陆，多数episode以crash或飘出视野终止——接近行为被距离惩罚驱动成功，但着陆信号过弱无法竞争。

`signal_completeness`：四个职责（goal_proximity、soft_landing、orientation、contact）形式上均存在，但proximity_reward作为无界状态值每步累计大额负奖励，淹没了着陆相关的speed_penalty_gated和contact_bonus；contact_bonus数学形态合理但因触发率极低（1%）无法提供有效的着陆引导。

`selected_level`：Level 2 — proximity_reward当前为无界状态值（每步-distant），满足"state_to_improvement"证据模式：agent在远处每步承受大额负奖励，靠近后信号强度急剧衰减，而着陆组件因相对尺度过小无法接管；仅调系数（Level 1）无法消除"待在远处就持续受罚"的结构性问题。

`selected_intervention`：唯一修改proximity_reward，从无界状态值`-distance`变为势能差分`2.0 * (distance - next_distance)`（正=靠近，负=远离）。其他三个组件（orientation_penalty、speed_penalty_gated、contact_bonus）完全不改动。

`falsifiable_hypothesis`：差分形式使每步proximity_reward幅度受物理步长约束（不再累积大额负数），靠近时获得正反馈，远离时受罚；当agent到达目标附近后proximity_reward趋近于0，speed_penalty_gated和contact_bonus的相对权重自然上升，着陆引导不再被淹没，从而减少逼近后crash的比例。

`expected_next_round`：proximity_reward的episode_sum_mean应从大额负数转为小额正数或零附近；其magnitude_share显著下降；speed_penalty_gated和contact_bonus的signed_share上升；score应改善（向0或正向移动）；early_terminal比例应下降。

`main_risk`：差分proximity不再直接奖励"保持在目标上"，若contact_bonus仍因触发率过低无法提供足够的停留激励，agent可能在到达目标附近后徘徊而不着陆——下一轮需关注contact_bonus的active_rate和episode_length是否异常增长。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft_contact, nright_contact = next_obs

    # ----------------------------------------------------------------
    # Core mandatory roles
    # ----------------------------------------------------------------
    # 1. Goal proximity: potential-based delta shaping
    #    Positive when moving toward target, negative when moving away
    distance = (x**2 + y**2)**0.5
    next_distance = (nx**2 + ny**2)**0.5
    proximity_reward = 2.0 * (distance - next_distance)

    # 2. Orientation penalty: keep body upright and stable
    orientation_penalty = -0.1 * (angle**2) - 0.1 * (angvel**2)

    # ----------------------------------------------------------------
    # Soft landing & settling (mandatory role)
    # ----------------------------------------------------------------
    # Gating factor: strong influence only when close to the target
    proximity_gate = 1.0 / (1.0 + 5.0 * distance)

    # Speed penalty: punish high velocities when near the pad
    speed_sq = vx**2 + vy**2
    speed_penalty_gated = -0.5 * speed_sq * proximity_gate

    # ----------------------------------------------------------------
    # Conditional role: safe contact bonus
    # ----------------------------------------------------------------
    # Reward stable two-leg contact when close to the pad
    contact_bonus = 0.5 * left_contact * right_contact * proximity_gate

    # ----------------------------------------------------------------
    # Combine components
    # ----------------------------------------------------------------
    total_reward = (
        proximity_reward +
        orientation_penalty +
        speed_penalty_gated +
        contact_bonus
    )

    components = {
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "speed_penalty_gated": speed_penalty_gated,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components
```