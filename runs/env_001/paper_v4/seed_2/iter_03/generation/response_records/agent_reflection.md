# Response Record

1. `evidence`：Score 220.24 已超 target 200（110%），20/20 正常终止、无截断，mean len 516；contact_completion episode_sum_mean=724.46（magnitude_share 65.8%）远大于 position_proximity=375.53（34.1%），velocity 和 angle 惩罚几乎为零；上一轮从 -17.90 跃升至 220.24 已解决任务。

2. `behavior_diagnosis`：Agent 已学会稳定着陆，但 episode 平均 516 步偏长，contact_completion 的 SUM 结构在无腿部接触时仍可通过位置/速度/角度因子获得 80% 最大奖励，可能鼓励智能体在着陆平台附近徘徊积累奖励后再完成最终触地。

3. `signal_completeness`：所有必要职责完备且可达 —— 位置引导、速度约束、姿态约束、接触确认均存在；contact_completion 的 SUM 稠密部分信用对学习有效，但幅度过大可能产生 hovering 局部最优。

4. `selected_level`：Level 1 —— contact_completion 的职责、符号、数学形态均正确，仅幅度相对 position_proximity 过强（约 1.93:1），降低系数是最小可验证干预。

5. `selected_intervention`：唯一修改目标组件 contact_completion，系数从 0.4 降至 0.2；其他三个组件完全不变。

6. `falsifiable_hypothesis`：contact_completion 每步奖励减半 → 在平台附近徘徊的边际收益下降 → episode length 缩短；同时 position_proximity 成为相对主导信号，驱使 Agent 更快接近原点完成着陆，外部 score 应维持或改善。

7. `expected_next_round`：episode length 应下降；score 保持 ≥ target；contact_completion episode_sum_mean 应按比例下降至 ~360；若 score 显著下降则说明原系数对着陆精度引导不可或缺。

8. `main_risk`：若 contact_completion 的实际贡献是精细着陆引导而非徘徊冗余，减半可能导致硬着陆或坠毁增加，外部 score 下降且方差扩大。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant states from the next observation (post-transition)
    x_next = next_obs[0]
    y_next = next_obs[1]
    vx_next = next_obs[2]
    vy_next = next_obs[3]
    angle_next = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- Component A: Position Proximity (Main Learning Signal) ----------
    # Dense, bounded reward encouraging the craft to reach (0,0).
    dist = (x_next**2 + y_next**2) ** 0.5
    pos_reward = 1.0 / (1.0 + dist)

    # ---------- Component B: Soft Landing Velocity (Bounded Penalty) ----------
    # Penalise high horizontal/vertical velocity only when close to the ground.
    activation = 1.0 / (1.0 + 10.0 * abs(y_next))
    vel_penalty = -0.1 * activation * (vx_next**2 + vy_next**2)

    # ---------- Component C: Stable Orientation (Quadratic Penalty) ----------
    # Light penalty on body tilt to encourage horizontal attitude.
    angle_penalty = -0.5 * (angle_next ** 2)

    # ---------- Component D: Contact Completion (Non-Collapsing Joint Sum) ----------
    # Each factor is an independent bounded [0,1] measure of a desired landing condition.
    # SUM ensures each condition provides its own gradient for partial credit.
    # Coefficient reduced from 0.4 to 0.2 to lessen incentive to loiter near the pad.

    k_x = 5.0
    factor_x = 1.0 / (1.0 + k_x * x_next**2)

    k_y = 5.0
    factor_y = 1.0 / (1.0 + k_y * y_next**2)

    k_v = 2.0
    factor_v = 1.0 / (1.0 + k_v * (vx_next**2 + vy_next**2))

    k_angle = 3.0
    factor_angle = 1.0 / (1.0 + k_angle * angle_next**2)

    factor_contact = 0.5 * left_contact + 0.5 * right_contact

    contact_reward = 0.2 * (factor_x + factor_y + factor_v + factor_angle + factor_contact)

    # ---------- Total Reward ----------
    total_reward = pos_reward + vel_penalty + angle_penalty + contact_reward

    components = {
        'position_proximity': pos_reward,
        'soft_landing_velocity': vel_penalty,
        'stable_orientation': angle_penalty,
        'contact_completion': contact_reward,
    }

    return float(total_reward), components
```
