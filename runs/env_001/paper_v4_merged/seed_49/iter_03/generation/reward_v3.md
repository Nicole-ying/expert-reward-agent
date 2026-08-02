1. `evidence`：所有 episode 在平均 68 步时因失败（坠毁/越界）终止，真实环境奖励每步约 -1.6156，而我设计的奖励每步仅 +0.0074，proxy 严重错位；soft_landing 占 signed_share 56.6% 但 active_rate 仅 1.2%，angle_penalty 完全无效（active_rate 0%），说明奖励函数缺少对失败状态的显式惩罚，且正向信号过弱无法对抗环境负奖励。

2. `behavior_diagnosis`：策略目前只会快速坠毁（~68 步内全部 early termination），没有学会减速或稳定姿态，progress 奖励鼓励了向目标移动却忽略了速度控制，导致高速撞击；真实得分持续极负且无改善。

3. `signal_completeness`：现有职责包括距离进步、姿态约束、软着陆、效率惩罚，但缺失对灾难性失败的直接惩罚信号（如越界、坠地），也缺乏终端成功奖励；angle_penalty 因阈值不当沦为僵尸组件，整体信号无法覆盖失败事件。

4. `selected_level`：Level 2，因观测到信号缺口（灾难性失败无覆盖），且上一轮尺度调整并未改变行为，符合“缺职责 → add 新组件”的结构变换条件。

5. `selected_intervention`：新增 `terminal_failure_penalty` 组件，当观测到失败状态（水平越界或垂直高度过低）时给予较大负惩罚，其余组件保持不变。

6. `falsifiable_hypothesis`：加入失败惩罚后，agent 在失败步骤将收到强负信号，从而迫使策略避免进入这些状态；预期训练初期的总 shaping reward 会更负，但随着学习，agent 会延长生存时间并可能逐步学会减速，最终真实得分应有所回升。

7. `expected_next_round`：`terminal_failure_penalty` 的 `active_rate` 将接近 1/len（仅最后一步触发），`episode_sum_mean` 会显著贡献负值（约 -5），总 shaping reward 均值将大幅下挫；但若训练有效，后续迭代的真实 score 应开始提高（或因惩罚过强导致策略瘫痪，表现更差）。

8. `main_risk`：惩罚值可能过大（-5）导致总奖励极度负面，使梯度消失或策略完全放弃运动，进而陷入更低分的静止策略；若发生，下一轮需降低惩罚系数或改用 gate 形式。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 权重与阈值
    w_progress = 1.0
    w_angle = 0.5
    w_angvel = 0.1
    w_soft_land = 2.0
    w_eff = 0.02
    w_failure = 5.0              # 失败惩罚权重

    angle_thresh = 0.3
    angvel_thresh = 1.0
    max_speed_land = 1.0
    max_angle_land = 0.5
    max_safe_vy = 0.5

    # 距离进展 + 安全下降门控
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    delta_dist = old_dist - new_dist

    vy = next_obs[3]
    downward_speed = -vy if vy < 0.0 else 0.0
    if downward_speed > max_safe_vy:
        overshoot = downward_speed - max_safe_vy
        gate = max(0.0, 1.0 - overshoot / max_safe_vy)
    else:
        gate = 1.0
    progress = w_progress * delta_dist * gate

    # 姿态稳定性惩罚
    angle = next_obs[4]
    angvel = next_obs[5]
    angle_penalty = -w_angle * max(0.0, abs(angle) - angle_thresh)
    angvel_penalty = -w_angvel * max(0.0, abs(angvel) - angvel_thresh)

    # 软着陆奖励（接触时）
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    any_contact = 1.0 if (left_contact + right_contact) > 0.5 else 0.0
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_factor = 1.0 - min(1.0, speed / max_speed_land)
    angle_factor = 1.0 - min(1.0, abs(angle) / max_angle_land)
    soft_landing_score = speed_factor * angle_factor
    soft_landing = w_soft_land * soft_landing_score * any_contact

    # 发动机使用惩罚
    eff_penalty = -w_eff * (0.0 if action == 0 else 1.0)

    # 新增：终端失败惩罚，在观测到失败状态时施加
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    # 失败条件：水平越界或高度过低（可能坠地/主体触地）
    if abs(x_pos) > 2.0 or y_pos < 0.1:
        failure_penalty = -w_failure
    else:
        failure_penalty = 0.0

    total_reward = (progress + angle_penalty + angvel_penalty +
                    soft_landing + eff_penalty + failure_penalty)

    components = {
        'progress': progress,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty,
        'soft_landing': soft_landing,
        'efficiency': eff_penalty,
        'failure_penalty': failure_penalty
    }
    return float(total_reward), components
```