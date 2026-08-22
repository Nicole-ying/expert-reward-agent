# Response Record

1. **evidence**：iter3 得分从 144.81 下滑至 141.28，episode 被 truncate 占 17/20，terminated 仅 3/20，agent 长时间（len≈898）维持接触但未能触发成功着陆；contact_reward 贡献 59.9% 份额，landing_progress 乘积可能塌缩且与原有 contact 奖励冲突，导致探索停滞。

2. **behavior_diagnosis**：agent 学会了长时间保持双脚接触（contact_reward 主导），但未完成"移动至中心、速度归零、姿态稳定并触发 settled 终止"的任务；它耗尽了步长上限，属于典型的“伪存活”策略。

3. **signal_completeness**：缺少一个在接近目标位形（中心、低速、低角速度、双脚接触）时提供强梯度的完成代理，当前 progress_delta 幅度极小不足以驱动精确着陆；安全约束阈值过高几乎不触发，无法防止未来可能的退化。

4. **selected_level**：Level 3 重建，因同骨架族（接触+进展+着陆乘积）连续三轮未刷新 best 且第三轮得分下降，满足"同骨架 ≥3 轮未刷新 best"的重建条件。

5. **selected_intervention**：移除持续 contact_reward 及乘积式 landing_progress，重建为 progress_delta (coefficient 5.0) + completion_proxy (几何平均) + 低阈值 hinge 安全惩罚；主信号由接近中心进展承担，任务完成信号通过多维联合条件在最终阶段给予强反馈。

6. **falsifiable_hypothesis**：放大 progress_delta 能推动 agent 更坚决地朝中心移动；以几何平均构建的 completion_proxy 在多条件接近理想值时给予正反馈，可诱导 agent 主动降低速度、稳定姿态并完成着陆，下一轮 score 应明显上升，episode_length 下降且 terminated 率提升。

7. **expected_next_round**：score 应提升至 160+，len 降至 800 以下，terminated 比例从 15% 升至 25%+；progress_delta episode_sum_mean 应 > 5，completion_proxy 出现非零活跃，contact_reward 被移除后不再主导奖励。

8. **main_risk**：若 progress_delta 系数过大，agent 可能贪功冒进导致 crash 率上升；几何平均因 contact 因子长时间为零导致 completion 信号迟迟不出现，早期完全依赖 progress_delta，可能减缓收敛。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft, nright = next_obs

    # Distances to target center (0,0)
    dist = (x**2 + y**2)**0.5 + 1e-8
    next_dist = (nx**2 + ny**2)**0.5 + 1e-8

    # 1. Progress towards center
    progress_delta = 5.0 * (dist - next_dist)

    # 2. Completion proxy (geometric mean of conditions)
    proximity = max(0.0, 1.0 - next_dist / 0.8)
    velocity_moderation = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 0.5)
    orientation_stability = max(0.0, 1.0 - abs(nangle) / 0.2)
    angvel_moderation = max(0.0, 1.0 - abs(nangvel) / 0.3)
    contact_factor = (nleft + nright) / 2.0

    proxy_product = proximity * velocity_moderation * orientation_stability * angvel_moderation * contact_factor
    completion_proxy = 1.0 * (proxy_product ** 0.2) if proxy_product > 0 else 0.0

    # 3. Safety penalties (hinge, low thresholds)
    speed_threshold = 0.4
    vx_violation = max(0.0, abs(nvx) - speed_threshold)
    vy_violation = max(0.0, abs(nvy) - speed_threshold)
    speed_penalty = -0.1 * (vx_violation + vy_violation)

    angle_threshold = 0.2
    angle_violation = max(0.0, abs(nangle) - angle_threshold)
    angle_penalty = -0.2 * angle_violation

    angvel_threshold = 0.3
    angvel_violation = max(0.0, abs(nangvel) - angvel_threshold)
    angvel_penalty = -0.1 * angvel_violation

    total_reward = progress_delta + completion_proxy + speed_penalty + angle_penalty + angvel_penalty

    components = {
        'progress_delta': progress_delta,
        'completion_proxy': completion_proxy,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total_reward), components
```
