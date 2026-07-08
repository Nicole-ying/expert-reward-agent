`evidence`：最终策略在全部20个episode中正常走到地形尽头（terminated=20/20，len≈1158），得分313.46。forward_reward占signed_share的76.8%，stability_penalty和energy_penalty分别占magnitude的14.9%和8.3%，比值均在合理范围，无组件塌缩或极端值支配。上一轮添加energy_penalty使得分从280.50提升至313.46，改善有效。

`behavior_diagnosis`：agent已经稳定完成行走任务（全部到达终点），但当前奖励完全缺失步态质量信号——腿接触信息（leg1_contact/leg2_contact）未被使用，agent可能以低效的拖步、双支撑过长或不对称步态完成行走，留下通过改善步态模式进一步降低能耗和稳定性惩罚的空间。

`signal_completeness`：前进引导（horizontal_velocity）、姿态约束（hull_angle + angular_vel + vertical_vel）、能耗约束（action magnitude）均已具备且可达。但缺少步态模式引导——双足行走的交替支撑相是高效行走的关键结构特征，当前obs提供了leg1_contact和leg2_contact但奖励未利用，属于可补充的过程引导信号。

`selected_level`：Level 2。当前组件职责基本完备且agent已能完成任务，但存在缺失的步态模式信号。添加基于接触的步态奖励属于新增组件（`independent_to_joint`方向的结构扩展），目的是在不破坏现有信号的前提下引入交替步态引导。

`selected_intervention`：新增`gait_reward`组件——当且仅当恰好一条腿接触地面时（单支撑相）给予小额正奖励，鼓励交替步态模式。保持forward_reward、stability_penalty、energy_penalty的系数和数学形态完全不变。

`falsifiable_hypothesis`：引入单支撑相奖励后，agent应发展出更清晰的交替迈步模式，减少双支撑拖步或无支撑跳跃时间。预期表现为episode_length略微缩短（步态更高效），energy_penalty和stability_penalty的绝对值下降，总得分提升。若干预无效，gait_reward的active_rate会接近100%但forward_reward和外部得分无实质改善（agent可能在原地交替抬腿而不前进）。

`expected_next_round`：gait_reward应具有高active_rate（步行中单支撑相频繁），episode_length可能从1158略降，stability_penalty和energy_penalty的magnitude_share下降，总score提升。terminated保持20/20。

`main_risk`：agent可能学会"原地踏步"——原地交替抬腿获取gait_reward而不前进。此时forward_reward会下降，总分不升反降，需在下一轮将gait_reward改为仅在有前进速度时生效的门控形式。

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
    w_angle = 1.0
    w_angvel = 0.2
    w_vert = 1.0
    w_energy = 0.05
    w_gait = 0.03

    forward_reward = w_forward * horizontal_velocity
    stability_penalty = (
        -w_angle * abs(hull_angle)
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