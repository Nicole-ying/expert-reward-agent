`evidence`：20/20 episodes全部以terminal结束且score<-50，平均长度仅68.8步；progress_reward（episode_sum_mean=1.118）与stability_penalty（-0.947）几乎完全抵消，|penalty/progress|=0.847远超0.5的经验触发器；soft_landing_proxy仅0.5%触发率，agent从未真正接近目标。

`behavior_diagnosis`：agent快速失败（crash或飞出边界），每步净奖励约+0.002几乎为零，梯度消失导致策略无法学习有意义的朝向目标运动。

`signal_completeness`：progress（delta距离）和stability（速度/角度/角速度惩罚）数学形态正确，符号合理，职责完备。问题不在信号缺失而在尺度失衡——惩罚过强使正向引导被淹没。

`selected_level`：Level 1。组件形态和职责未被证伪，证据明确指向stability_penalty系数过强（|ratio|>0.5触发），只需尺度修复。

`selected_intervention`：单一目标组件`stability_penalty`，系数降低约10倍：w_vel 0.01→0.001，w_angle 0.01→0.001，w_angvel 0.001→0.0001。

`falsifiable_hypothesis`：惩罚系数降低10倍后，预期|penalty/progress|降至~0.08-0.12区间，progress成为主导信号，agent获得清晰的朝向目标的正梯度，应能学会朝目标移动而非原地振荡后crash。

`expected_next_round`：stability_penalty的magnitude_share显著下降；progress_reward signed_share上升至80%以上；episode_length可能先波动后增长（探索更多动作）；score应从-100.5改善。

`main_risk`：惩罚过弱可能导致agent不学习减速和姿态控制，到达目标区域后因速度过高而crash。若下一轮出现"到达但无法着陆"行为，需通过training_progress调度或门控恢复后期约束。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: Scale fix — stability penalty coefficients reduced ~10x to stop
    cancelling the progress signal. Progress and soft_landing_proxy unchanged.
    """
    # -- Helper: distance to target (target is at (0,0))
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # -- 1. Progress delta (main driving signal, unchanged)
    delta_dist = dist_current - dist_next
    w_progress = 1.0
    progress_reward = w_progress * delta_dist

    # -- 2. Stability penalty: coefficients reduced ~10x
    w_vel   = 0.001   # was 0.01
    w_angle = 0.001   # was 0.01
    w_angvel= 0.0001  # was 0.001

    stability_penalty = (
        -w_vel   * (abs(next_obs[2]) + abs(next_obs[3]))
        -w_angle * abs(next_obs[4])
        -w_angvel* abs(next_obs[5])
    )

    # -- 3. Soft landing proxy (unchanged)
    dist_threshold   = 0.5
    vel_threshold    = 0.3
    angle_threshold  = 0.2

    near_target = dist_next < dist_threshold
    low_speed   = (abs(next_obs[2]) < vel_threshold) and (abs(next_obs[3]) < vel_threshold)
    upright     = abs(next_obs[4]) < angle_threshold
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)

    w_proxy = 0.5
    soft_landing_proxy = w_proxy if (near_target and low_speed and upright and both_contact) else 0.0

    # -- Total reward
    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```