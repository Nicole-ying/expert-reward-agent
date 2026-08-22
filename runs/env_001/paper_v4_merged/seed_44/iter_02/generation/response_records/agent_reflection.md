# Response Record

1. `evidence`：评估分-18.89，全部20回合截断（1000步），无早期终止，contact_bonus与speed_penalty激活率0%，progress正向但很小（12.09），angle_penalty持续负向（-2.65），说明策略徘徊未着陆。
2. `behavior_diagnosis`：智能体学会在目标周围存活1000步而不坠毁，但因稀疏contact_bonus从未触发，无法学会软着陆，形成hovering proxy。
3. `signal_completeness`：接近信号（progress）正常，姿态约束（angle）持续但不可达，缺少着陆过程的连续引导信号来对齐任务完成。
4. `selected_level`：Level 2 — contact_bonus为僵尸组件（active_rate 0%），需结构变换从稀疏二值改为连续稠密的着陆引导。
5. `selected_intervention`：删除contact_bonus，替换为连续landing_reward（距离衰减×低速因子），系数0.01以保持尺度服从。
6. `falsifiable_hypothesis`：连续landing_reward将enable着陆引导，landing_reward激活率应>0且评估分上升，若上升不明显则需进一步调整姿态约束或引入成功终止信号。
7. `expected_next_round`：landing_reward active_rate > 20%，评估分提升至-15以上，episode长度可能仍为1000但最终接近垫面次数增加。
8. `main_risk`：landing_reward鼓励低速可能导致在较高位置停留（hover），因为距离因子仍可提供一定奖励，需后续加入高度惩罚或下降奖励。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前位置与目标垫距离
    x_curr = obs[0]
    y_curr = obs[1]
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5

    x_next = next_obs[0]
    y_next = next_obs[1]
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # 主进展信号：到目标的欧氏距离递减
    progress = dist_curr - dist_next

    # 下一步速度（用于原有速度约束）
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]

    # 速度硬约束（保留但可能保持僵尸）
    x_speed_viol = max(0.0, abs(x_vel_next) - 0.8)
    y_speed_viol = max(0.0, -y_vel_next - 0.8)
    speed_penalty = x_speed_viol + y_speed_viol

    # 姿态与角速度稳定约束
    body_angle_next = next_obs[4]
    angular_vel_next = next_obs[5]
    angle_penalty = body_angle_next ** 2 + angular_vel_next ** 2

    # 替换 contact_bonus：连续软着陆引导
    # 距离因子：越近奖励越大（指数衰减，半衰距离0.5）
    dist_factor = 2.718281828 ** (-dist_next / 0.5)
    # 速度因子：水平与垂直线速度绝对值和越小奖励越大（线性衰减，1.0处归零）
    speed_factor = max(0.0, 1.0 - (abs(x_vel_next) + abs(y_vel_next)) / 1.0)
    landing_reward = dist_factor * speed_factor

    # 加权组合
    total = (
        10.0 * progress
        - 1.0 * speed_penalty
        - 0.5 * angle_penalty
        + 0.01 * landing_reward
    )

    components = {
        "progress": 10.0 * progress,
        "speed_penalty": -1.0 * speed_penalty,
        "angle_penalty": -0.5 * angle_penalty,
        "landing_reward": 0.01 * landing_reward
    }

    return float(total), components
```
