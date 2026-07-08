## 诊断与修改

**evidence**：score=-108.37（new best，较上轮+4.59），episode_length=68.45且20/20全部early terminal（<150步, score<-50）；distance_reward=-66.46（79% magnitude），landing_quality=9.46（42.4% active），stability_penalty=-8.17（100% active）；上一轮将product改为geometric mean后landing_quality从0.010跃升至0.139，得分小幅改善，但崩溃模式未变。

**behavior_diagnosis**：agent在约68步时全部提前终止（crash或飞出边界），从未成功着陆。稳定性惩罚对速度、倾角、角速度持续施加约束，可能阻碍agent做出接近目标和着陆所需的大幅度控制动作，导致agent在缓慢调整过程中撞地或越界。

**signal_completeness**：distance_reward提供到达目标的稠密梯度，landing_quality在2单位距离内提供着陆引导，stability_penalty提供安全约束。但任务明确要求的"节省发动机推力"信号缺失，且稳定性约束在早期探索阶段可能过强。搜索知识库确认early_failure_or_crash模式建议"大幅降低稳定性惩罚"。

**selected_level**：Level 1 —— 稳定性惩罚的职责和数学形态合理，但尺度过强阻碍早期探索和必要机动。上一轮修改了landing_quality形态（已生效），本轮不应重复修改同一组件；stability_penalty系数尚未被调整过，且ratio虽低于0.5但处于crash场景下的"过度约束"状态。

**selected_intervention**：仅调整stability_penalty的四个权重系数，将其整体幅度降至约40%（w_vx: 0.15→0.06, w_vy: 0.05→0.02, w_angle: 0.2→0.08, w_angvel: 0.2→0.08）。distance_reward和landing_quality保持不变。

**falsifiable_hypothesis**：降低稳定性约束后，agent能做出更大幅度必要机动（调整姿态、减速、对准平台），episode应显著延长（存活更久），score应改善（减少因无法机动而撞地的惩罚积累），stability_penalty的episode_sum_mean绝对值应成比例下降。

**expected_next_round**：episode_length > 80，score > -100，early_terminal比例下降，stability_penalty magnitude降至约-3到-5。

**main_risk**：稳定性约束过弱可能导致agent学习到剧烈摆动或不稳定策略，产生新的failure模式（如高速旋转后甩出边界）。若下一轮尺度异常已消失但行为无实质改善，则转Level 2添加安全接近引导信号或燃料代价。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observables
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Primary learning signal: continuous negative Euclidean distance to goal
    distance_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -1.0 * distance_to_target

    # 2. Stability penalty: significantly reduced to avoid hindering necessary maneuvers
    #    Previous weights (0.15, 0.05, 0.2, 0.2) overly constrained early exploration.
    w_vx = 0.06
    w_vy = 0.02
    w_angle = 0.08
    w_angvel = 0.08
    stability_penalty = -(
        w_vx * abs(x_vel) +
        w_vy * abs(y_vel) +
        w_angle * abs(body_angle) +
        w_angvel * abs(angular_vel)
    )

    # 3. Gradual landing quality: geometric mean of bounded factors
    prox_factor = max(0.0, 1.0 - distance_to_target / 2.0)
    speed_x_factor = max(0.0, 1.0 - abs(x_vel) / 0.8)
    speed_y_factor = max(0.0, 1.0 - abs(y_vel) / 0.8)
    angle_factor = max(0.0, 1.0 - abs(body_angle) / 0.4)
    contact_factor = 0.2 + 0.8 * (left_contact + right_contact) / 2.0

    product_of_factors = prox_factor * speed_x_factor * speed_y_factor * angle_factor * contact_factor
    landing_quality = 0.8 * (product_of_factors ** 0.2)

    total_reward = distance_reward + stability_penalty + landing_quality

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```