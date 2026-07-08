# Response Record

`evidence`：上一轮新增的gait_reward对分数几乎无影响（+0.27），forward_reward占据80.2% signed_share，stability_penalty使用abs(hull_angle)线性形式，对正常步态微小晃动与危险倾斜施加相同梯度，无法区分严重程度。

`behavior_diagnosis`：策略已能走完全程（terminated=20/20，score≈314），但速度与效率进入平台期；线性稳定性惩罚对步态中必然出现的±0.03~0.05弧度身体摇摆与真正危险的倾斜施加相同比例的惩罚，可能抑制了更快速、更自然的步态探索。

`signal_completeness`：前进引导、稳定性约束、能量约束、步态塑造均已存在，职责完备可达；但稳定性分量的线性abs形态缺乏对微小振荡与危险倾斜的梯度区分。

`selected_level`：Level 2。线性abs在接近零处梯度恒为±1，缺乏对不同幅度倾斜的差异化反馈，匹配`state_to_improvement`方向——将"状态绝对值"改为"状态平方"，使梯度随倾斜幅度线性增长。

`selected_intervention`：唯一修改stability_penalty中的hull_angle项——将`abs(hull_angle)`替换为`hull_angle ** 2`，并将w_angle从1.0调整至20.0，使在典型步态振荡幅度（~0.05 rad）处的惩罚与原线性形式持平，但对危险倾斜（>0.2 rad）的惩罚急剧放大。

`falsifiable_hypothesis`：平方形式对正常步态振荡（|hull_angle|<0.05）更宽容，允许策略探索更大步幅和更快速度；对危险倾斜的强梯度应保持或改善稳定性。结果应为episode_length缩短（走得更快）或forward_reward均值上升，同时stability_penalty不显著恶化。

`expected_next_round`：forward_reward均值应上升（速度更快），stability_penalty magnitude可能略微下降（小幅振荡不再被同等惩罚），score应超越当前best 313.73，episode_length可能缩短。

`main_risk`：w_angle=20.0若过高，可能使整体stability_penalty膨胀，压制前进激励导致策略趋于保守不动；若出现此情况，下一轮应将w_angle降至5.0~10.0。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant next observations
    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    horizontal_velocity = next_obs[2]
    vertical_velocity = next_obs[3]
    leg1_contact = next_obs[8]
    leg2_contact = next_obs[13]

    # Weights
    w_forward = 1.0
    w_angle = 20.0
    w_angvel = 0.2
    w_vert = 1.0
    w_energy = 0.05
    w_gait = 0.03

    forward_reward = w_forward * horizontal_velocity
    stability_penalty = (
        -w_angle * (hull_angle ** 2)
        - w_angvel * abs(hull_angular_velocity)
        - w_vert * (vertical_velocity ** 2)
    )
    energy_penalty = -w_energy * (
        action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2
    )
    # Single-leg support reward: exactly one foot on ground
    single_leg_contact = float(abs(leg1_contact - leg2_contact) > 0.5)
    gait_reward = w_gait * single_leg_contact

    total_reward = forward_reward + stability_penalty + energy_penalty + gait_reward
    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty,
        "gait_reward": gait_reward
    }

    return float(total_reward), components
```
