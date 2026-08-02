# Response Record

1. `evidence`：本轮best score从-87.19刷新至-24.05，len从143暴涨至980.75，truncated 19/20说明agent几乎全程存活但未完成任务；success_bonus提供+75.4正向信号，而progress及shaped_progress几乎为零（1.19/0.41），表明agent利用姿态与接触获取奖励却无目标接近行为；gate_angle与contact_factor统计值虚高（因子而非奖励）但反映了姿态保持行为。
2. `behavior_diagnosis`：agent学会了通过保持竖直、双脚触地来持续领取success_bonus，同时回避任何向目标的有效移动，形成“存活但无进展”的exploit，导致进度信号瘫痪。
3. `signal_completeness`：已声明且可用的距离、速度、角度、接触均被使用，但缺少与进度绑定的正向完成信号；现有success_bonus为纯状态奖励，缺失“进步才能获奖”的职责。
4. `selected_level`：Level 2 — 结构变换，触发条件为“占据好状态即持续获奖”，需将success_bonus从state改换为improvement。
5. `selected_intervention`：移除基于姿态–距离几何平均的success_bonus，替换为仅依赖位移进度的progress_bonus = constant * max(0, delta_dist)。
6. `falsifiable_hypothesis`：若进度奖励取代姿态奖励，agent将被迫向目标移动以获得正向信号，下一轮progress_bonus应成为最大正贡献项且score随len缩短而上升，反之若score不变或下降则说明缺少姿态约束导致频繁坠毁或出界。
7. `expected_next_round`：progress_bonus episode_sum_mean > 20，len降至400以下，terminated比例上升，score区间上移且至少一个episode score > 50；进度相关组件总和占比超过30%。
8. `main_risk`：缺乏姿态门控可能导致agent高速撞地或飞出视口而快速terminate，score可能进一步恶化。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nl_contact, nr_contact = next_obs

    # distances to target (0,0)
    dist = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5

    # 1. progress delta: positive when approaching target
    delta_dist = dist - dist_next
    progress = 1.0 * delta_dist

    # 2. angle gate: linearly decay progress when body angle exceeds safe range
    safe_angle = 0.5  # radians
    gate_angle = max(0.3, 1.0 - abs(nangle) / safe_angle)

    # 3. contact factor: encourage both legs on ground
    if nl_contact == 1 and nr_contact == 1:
        contact_factor = 1.0
    elif nl_contact == 1 or nr_contact == 1:
        contact_factor = 0.7
    else:
        contact_factor = 0.4

    # shaped progress: main learning signal with safety and contact modulation
    shaped_progress = progress * gate_angle * contact_factor

    # 4. speed penalty near ground to promote gentle landing
    close_threshold = 0.5
    speed_penalty = 0.0
    if dist_next < close_threshold:
        speed_norm = abs(nvx) + abs(nvy)
        speed_penalty = -0.1 * speed_norm

    # 5. progress-based bonus: reward only net approach, replacing static success_bonus
    progress_bonus = 5.0 * max(0.0, delta_dist)

    # 6. action cost: small penalty for any engine use
    action_cost = -0.01 if action != 0 else 0.0

    total_reward = shaped_progress + progress_bonus + speed_penalty + action_cost

    components = {
        "progress": progress,
        "gate_angle": gate_angle,
        "contact_factor": contact_factor,
        "shaped_progress": shaped_progress,
        "speed_penalty": speed_penalty,
        "progress_bonus": progress_bonus,
        "action_cost": action_cost
    }
    return float(total_reward), components
```
