1. evidence：最终策略得分仅 142.74，88.5% 的 episode 时间（mean len 885）在超时中徘徊，landing_bonus 和 contact_reward 占据奖励主份额（signed_share 50%+47%），而 progress_delta 仅贡献 1.7%，说明 agent 因持续触发接触奖励而在平台区域停留，未真正完成稳定着陆；历史最佳骨架 iter4 使用 completion_proxy + progress_delta + hinge penalties 曾达到 195.26，连续 3 轮（iter5‑7）未超过该 best。

2. behavior_diagnosis：agent 利用 landing_bonus 和 contact_reward 在目标区域反复获得正向反馈，但并未收敛到双脚稳定接触、低速低角度的到位状态，所有回合均超时截断，缺失成就感信号致使策略陷入“靠近但不完成”的 exploit。

3. signal_completeness：主进展信号（距离缩短）、完成信号（可通过距离、速度、角度、角速度、接触构造）和安全约束（速度、姿态）均可用，但上一轮缺失完成导向，仅靠状态奖励无法区分“靠近”和“完成”。

4. selected_level：Level 3 重建。触发条件：同一骨架族连续 3 轮未刷新 best（iter4 以来），且上一轮 contact/landing 结构明显 exploit，需更换主信号框架。

5. selected_intervention：全新骨架以 progress_delta + completion_proxy（乘积式 joint condition）+ speed/angle/angvel hinge penalties；completion_proxy 由距离、速度、角度、角速度、双足接触五个连续因子相乘后乘以 15.0，形成仅在接近完美着陆时骤升的强完成梯度，progress_delta 提供持续靠近中心的基础信号。

6. falsifiable_hypothesis：completion_proxy 的急剧上升将要求同时满足小距离、低速度、小姿态角、低角速度和双足接触，从而推动 agent 从“靠近徘徊”切换为“精确柔和完成”，预期 score 将接近或突破前最佳 195.26。

7. expected_next_round：score > 180，episode_length 降至 600‑800，completion_proxy 组件 active_rate > 50% 且 episode_sum_mean 显著提升，不存在 landing_bonus/contact_reward 类 exploit 组件。

8. main_risk：completion_proxy 因其乘积结构在未完全满足时可能长时间为 0，导致稀疏反馈，agent 探索不足从而陷入更差的徘徊或 crash；若出现此现象，下一步应引入 soft_health_gate 或更稠密的 proxy 边界调整。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft, nright = next_obs

    # ----- 1. 主进展信号：向中心靠近 -----
    dist = (x**2 + y**2)**0.5 + 1e-8
    next_dist = (nx**2 + ny**2)**0.5 + 1e-8
    progress = 5.0 * (dist - next_dist)   # 正奖励=靠近中心

    # ----- 2. 完成信号（乘积式 joint condition proxy）-----
    # 每个因子在条件未满足时为 0，满足时线性趋近 1
    proximity      = max(0.0, 1.0 - next_dist / 0.3)          # 距离 <0.3 时才有正贡献
    velocity_ok    = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 0.5)  # 联合线速度 <0.5 时非零
    angle_ok       = max(0.0, 1.0 - abs(nangle) / 0.2)        # 倾角 <0.2 rad
    angvel_ok      = max(0.0, 1.0 - abs(nangvel) / 0.3)       # 角速度 <0.3
    contact_ok     = (nleft + nright) / 2.0                   # 双脚接触=1, 单脚=0.5, 无=0

    completion = proximity * velocity_ok * angle_ok * angvel_ok * contact_ok
    completion_bonus = 15.0 * completion   # 只有全部条件同时接近满足时才大

    # ----- 3. 安全约束（hinge 惩罚，仅在超出阈值时生效）-----
    speed_penalty  = -0.3 * (max(0.0, abs(nvx) - 0.5) + max(0.0, abs(nvy) - 0.5))
    angle_penalty  = -0.6 * max(0.0, abs(nangle) - 0.2)
    angvel_penalty = -0.2 * max(0.0, abs(nangvel) - 0.5)

    total_reward = progress + completion_bonus + speed_penalty + angle_penalty + angvel_penalty

    components = {
        'progress': progress,
        'completion_bonus': completion_bonus,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total_reward), components
```