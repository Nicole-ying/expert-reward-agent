# Response Record

`evidence`: 当前score=150.95（目标75%），terminated=16/20，episode_length=796偏长；landing_quality占signed_share的93.9%但active_rate仅4.5%，progress的active_rate为98.5%但episode_sum_mean仅1.39，说明长达~760步的接近阶段几乎无有效学习信号，而~36步的接触阶段主导了全部奖励。

`behavior_diagnosis`: agent能够着陆（16/20自然终止），但接近阶段效率极低——progress均值1.39远低于初始距离可提供的上限，策略在接近过程中徘徊或振荡，导致episode过长、部分episode得分极低（最低-18.63），拉低了均值。

`signal_completeness`: 职责基本完备——progress提供接近引导、orientation_penalty维持姿态稳定、landing_quality覆盖软着陆条件。但progress信号虽活跃（98.5%），其幅值塌缩到几乎为零，无法在接近阶段提供有效梯度。其他职责形态合理。

`selected_level`: Level 1。progress组件的职责、符号和数学形态正确（逐步差分引导接近），证据仅表明其相对于landing_quality尺度过弱，导致接近阶段信用分配不足。先尝试系数修复。

`selected_intervention`: 唯一目标组件为progress，将其系数w_progress从1.0提升至5.0，其他组件不变。

`falsifiable_hypothesis`: 更强的progress信号应在接近阶段（占98.5%步数）提供有意义的梯度，促使agent减少徘徊、更直接地飞向目标平台，从而缩短episode长度、提高接近效率，并为landing_quality的触发创造更多机会；平均得分应因此上升。若下一轮progress的episode_sum_mean成比例放大但episode_length和得分无实质改善，则说明问题不在尺度而在数学形态。

`expected_next_round`: progress的episode_sum_mean应从~1.39升至~5-10（取决于行为改善程度），episode_length应从796下降，score应上升，truncated比例应下降。

`main_risk`: 若agent在接近阶段有振荡（正负progress交替），放大系数会同时放大正负波动，可能导致少数episode因大幅负progress而得分更差，极端情况下可能促使agent采取更冒险的快速下降策略导致crash增加。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # weights and sensitivities
    w_progress = 5.0
    w_angle_penalty = 0.2
    w_landing = 1.0
    a_v = 10.0       # sensitivity for vertical speed
    b_angle = 10.0   # sensitivity for body angle in landing quality

    # current and next distances to target (0,0)
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 1. Main progress signal: improvement towards target
    progress = w_progress * (dist - next_dist)

    # 2. Orientation stability: penalty for body angle to prevent tumbling and crash
    orientation_penalty = -w_angle_penalty * (next_obs[4] ** 2)

    # 3. Landing quality bonus: soft success proxy when contacts are active
    contact = max(next_obs[6], next_obs[7])  # any leg contact
    vertical_speed_factor = 1.0 / (1.0 + a_v * (next_obs[3] ** 2))
    angle_factor = 1.0 / (1.0 + b_angle * (next_obs[4] ** 2))
    landing_quality = w_landing * contact * vertical_speed_factor * angle_factor

    total_reward = progress + orientation_penalty + landing_quality

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```
