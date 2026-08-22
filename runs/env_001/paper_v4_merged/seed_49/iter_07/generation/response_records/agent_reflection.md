# Response Record

1. `evidence`：当前评估中 19/20 episode 被时间截断，仅 1 个 terminated，episode 平均长度高达 956；progress 组件占据 98.5% 的 magnitude_share，overspeed_penalty 仅 3.7% 激活，说明 agent 通过极慢速移动积累 progress 奖励导致拖延，缺少完成信号。
2. `behavior_diagnosis`：agent 学会悬停或极慢漂移接近原点，每步赚取微小 progress，但从未完成真正着陆，直至被截断，因此任务完成率几乎为零。
3. `signal_completeness`：任务必需的完成奖励完全缺失，姿态、速度、接触传感器未被用于构造着陆完成条件，导致 progress proxy 被 exploit。
4. `selected_level`：Level 2（proxy→completion_alignment），根据 progress 提高而外部分数未升（terminated 率低）的证据触发的结构变换。
5. `selected_intervention`：新增 `success_bonus` 组件，在满足推断成功着陆条件时给予 +200 奖励，保留现有 progress 与 overspeed_penalty。
6. `falsifiable_hypothesis`：加入成功奖励后，agent 应有动机在接近原点时完成着陆，从而 terminated 率上升、episode 长度下降，并且 progress exploit 导致的长期徘徊会被抑制。
7. `expected_next_round`：terminated 比例应 >20%，episode 平均长度降至 500 以下，`success_bonus` 的 active_rate 大于 0，agent 的得分可能略有波动但最终会受成功奖励主导。
8. `main_risk`：阈值过严可能导致 success_bonus 始终不触发，agent 继续无目标徘徊；过松可能让坠落接触被误判为成功，引入错误引导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 1) Progress: distance reduction ----------
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    delta_dist = dist - next_dist
    progress = 1.0 * delta_dist

    # ---------- 2) Overspeed penalty ----------
    speed = (vx ** 2 + vy ** 2) ** 0.5
    safe_speed = 0.5
    overspeed = max(0.0, speed - safe_speed)
    overspeed_penalty = -0.005 * overspeed

    # ---------- 3) Landing success bonus ----------
    # Conditions: at least one leg contact, very close to origin,
    # very low speed, and upright body angle.
    contact = (left_contact > 0.5) or (right_contact > 0.5)
    dist_small = next_dist < 0.2
    speed_low = speed < 0.2
    angle_ok = abs(body_angle) < 0.3

    success_bonus = 200.0 if (contact and dist_small and speed_low and angle_ok) else 0.0

    # ---------- Total reward ----------
    total_reward = progress + overspeed_penalty + success_bonus

    components = {
        'progress': progress,
        'overspeed_penalty': overspeed_penalty,
        'success_bonus': success_bonus
    }
    return float(total_reward), components
```
