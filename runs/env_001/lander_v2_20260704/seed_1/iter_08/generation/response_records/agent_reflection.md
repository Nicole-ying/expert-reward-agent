# Response Record

**evidence**: iter 7的state-based approach_signal以72.4% signed share和100% active_rate主导奖励，而landing_bonus仅2.7%触发率；best代码（iter 5, 108.45分）使用improvement-based结构，当前版本是从improvement向state-based的结构性回退。

**behavior_diagnosis**: 智能体在着陆垫附近徘徊收割approach_signal的持续奖励，却极少完成着陆——episode长达623.7步、得分-79.47、landing_bonus触发率仅2.7%，形成典型的"接近但不降落"徘徊行为。

**signal_completeness**: best代码的改进量奖励覆盖了接近、减速、姿态和触地的过程引导，但缺失终端状态确认信号——一旦满足着陆条件，所有改进量归零，智能体在成功状态得不到正向强化，只能依赖环境自发终止。

**selected_level**: Level 2——知识库命中goal_near_oscillation模式（landing proxy触发率<10%），且best代码的纯改进量结构在终端状态存在信号塌缩，需要添加bounded连续proxy来填补终端确认职责。

**selected_intervention**: 在approach_landing_reward内部新增landing_state_reward——当双腿触地、接近垫中心、低速且直立时持续发放有界乘积型奖励，为维持着陆状态提供正向梯度。

**falsifiable_hypothesis**: 终端状态持续确认信号应使智能体从"接近后徘徊"转向"接近后着陆并保持"，表现为episode_length缩短、terminated比例保持高位、且新增landing_state_reward有>10%的active_rate。

**expected_next_round**: score应回到100+区间，episode_length从623降至300-500，landing_state_reward的active_rate应明显高于iter 7 landing_bonus的2.7%。

**main_risk**: 若gate条件过松，智能体可能在单腿触地、不完全稳定的状态下收割landing_state_reward；当前乘积门控使用有理函数（非指数）提供更平滑的梯度，但若出现该类proxy farming需收紧gate参数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取当前和下一步状态
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离与稳定性成本
    w_dist = 0.08
    w_vel = 0.1
    w_angle = 0.1
    w_angvel = 0.05

    distance_cost = -w_dist * (nx ** 2 + ny ** 2)
    stability_cost = -(
        w_vel * (vx ** 2 + vy ** 2)
        + w_angle * abs(angle)
        + w_angvel * abs(omega)
    )

    # 改善量型 approach_landing_reward
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    curr_dist = (nx ** 2 + ny ** 2) ** 0.5
    prev_vel = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    curr_vel = (vx ** 2 + vy ** 2) ** 0.5
    prev_contact = (obs[6] + obs[7]) / 2.0
    curr_contact = (left_contact + right_contact) / 2.0

    # 接近改善：向目标靠近即获奖
    proximity_improvement = 5.0 * max(0.0, prev_dist - curr_dist)

    # 减速改善：仅在靠近着陆垫时奖励减速
    proximity_gate = max(0.0, 1.0 - curr_dist / 1.5)
    velocity_improvement = 10.0 * proximity_gate * max(0.0, prev_vel - curr_vel)

    # 姿态改善：减小倾角即获奖
    angle_improvement = 3.0 * max(0.0, abs(obs[4]) - abs(angle))

    # 触地改善：建立支撑腿接触的转移奖励
    contact_improvement = 30.0 * max(0.0, curr_contact - prev_contact)

    # 终端着陆状态持续确认：双腿触地、近中心、低速、直立的乘积型有界奖励
    # 填补改进量在着陆完成后归零的信号缺口
    landing_state_reward = 3.0 * (
        (left_contact + right_contact) / 2.0
        * (1.0 / (1.0 + 5.0 * nx ** 2))
        * (1.0 / (1.0 + 5.0 * ny ** 2))
        * (1.0 / (1.0 + 3.0 * (vx ** 2 + vy ** 2)))
        * (1.0 / (1.0 + 1.0 * angle ** 2))
    )

    approach_landing_reward = (
        proximity_improvement
        + velocity_improvement
        + angle_improvement
        + contact_improvement
        + landing_state_reward
    )

    total_reward = distance_cost + stability_cost + approach_landing_reward

    components = {
        'distance_cost': distance_cost,
        'stability_cost': stability_cost,
        'approach_landing_reward': approach_landing_reward,
    }

    return float(total_reward), components
```
