# Response Record

`evidence`：当前event_bonus以10.0一次性事件主导奖励（signed_share=66.1%），但active_rate仅8.0%，且触发条件为"接触状态转移"，这鼓励反复弹跳重新触发而非稳定停靠；best代码用连续state-based settlement_bonus避免了此问题，得分187.04远优于当前168.77。

`behavior_diagnosis`：agent学会在平台上弹跳——每次双腿触地+低速+直立就赚取10.0事件奖励，然后弹起失去接触，再落下重新触发，反复循环却未真正settle；episode较短（517 vs best的715）且terminated=15/20说明最终确实触发了环境终止，但弹跳过程浪费了大量步数且event_bonus与实际任务完成脱节。

`signal_completeness`：best代码的信号职责完备——potential-based proximity引导接近、orientation_penalty维持姿态、speed_penalty_gated在平台附近减速、continuous settlement_bonus鼓励保持settled状态。唯一可改进的是settlement_bonus使用`nleft_contact * nright_contact`乘积门控，单脚接触时梯度为零。

`selected_level`：Level 2。best的乘积接触门控使单脚接触时settlement_bonus=0，形成奖励悬崖；改为平均接触提供平滑梯度，属于门控结构的数学变换。

`selected_intervention`：以best代码为基础，仅修改settlement_bonus的接触因子，从`nleft_contact * nright_contact`（二元乘积）改为`(nleft_contact + nright_contact) / 2.0`（连续平均），其他组件和系数完全保持best原样。

`falsifiable_hypothesis`：平均接触为单脚触地提供0.5倍部分奖励，消除"无接触到双脚接触"之间的奖励悬崖，使agent能更平滑地学会从单脚支撑过渡到双脚稳定settle，减少弹跳试探次数，提高settle效率，从而将分数从187推向200+。

`expected_next_round`：settlement_bonus的active_rate应上升（从best可能的低值），episode_length可能略降（更快settle），score应突破best的187.04向200靠近，terminated比例应保持高位。

`main_risk`：单脚接触的部分奖励可能让agent满足于单脚悬停（farm 1.0/step而非2.0/step），若proximity_gate和stillness不够严格，可能出现"单脚点地、低速维持"但不真正settle的中间策略。

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
    # 4. Settlement bonus: continuous reward for settled state.
    #    CHANGED from product contact (binary 0/1) to average contact
    #    (continuous 0.0/0.5/1.0) so single-foot contact gets partial
    #    credit, eliminating the reward cliff between one and two feet.
    # ----------------------------------------------------------------
    next_proximity_gate = 1.0 / (1.0 + 5.0 * next_distance)
    next_speed_sq = nvx**2 + nvy**2
    stillness = 1.0 / (1.0 + 10.0 * (next_speed_sq + nangvel**2))
    # Average contact: 0.0 none, 0.5 one foot, 1.0 both feet
    avg_contact = (nleft_contact + nright_contact) / 2.0
    settlement_bonus = 2.0 * avg_contact * next_proximity_gate * stillness

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
