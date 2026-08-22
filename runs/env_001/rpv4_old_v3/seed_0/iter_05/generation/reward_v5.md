1. evidence：历史 iter 2 得分 217（已达标），其 landing_success_reward 为二元正奖励（active_rate 34.8%），当前 iter 4 因 living_penalty 加倍导致 len 骤降、score 崩塌。上一次被拒草案实质重复 iter 4，非有效干预；必须从 iter 2 代码出发做截然不同的修改。
2. behavior_diagnosis：策略当前过早终止，回合长度仅 109 步，大量以 early termination（score<-50）结束，说明过重的时间惩罚迫使 agent 未完成着陆即崩溃；而在 iter 2 中 agent 已学会抵达目标并执行着陆动作，证明主引导结构有效。
3. signal_completeness：现有组件已覆盖接近引导、着陆判定和轻量时间约束，binary landing reward 提供稀疏完成激励；为增强着陆过程的稠密引导，可将其升级为与距离、速度、姿态、双腿接触程度成比例的连续软着陆奖励，信号仍充分且可达。
4. selected_level：Level 2（修改组件数学形式）。触发条件：landing_success_reward 为二元开关信号，改为连续软着陆奖励，提供更细粒度的中间激励。
5. selected_intervention：唯一目标组件为 `landing_success_reward`，将其替换为基于门控乘积的连续正奖励 `soft_landing_bonus`，同时保留 iter 2 中几乎无副作用的 crash_prevention 和 living_penalty=-0.002；移除原二元着陆奖励。
6. falsifiable_hypothesis：连续软着陆奖励在靠近目标垫、减速、摆正姿态且双腿接触时给出梯度回报，应帮助 agent 更稳定地学会着陆而不依赖单一稀疏事件，在保持或提升分数的同时避免 living_penalty 主导导致的过早终止。
7. expected_next_round：score 应回升至 200 以上，len 增加至 300+，terminated 比例从 100% 下降（部分因着陆失败而非 early termination 结束），landing_success_reward 消失而被 soft_landing_bonus 替代，active_rate 高。
8. main_risk：连续奖励可能使 agent 满足于单腿接触或低速盘旋而未完全稳定着陆，导致不完整着陆时的 crash 增多；需通过后续轮次观察是否需要收紧 contact 条件或降低权重。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离计算
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（距离改善差分）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)

    # 2. 软着陆引导：连续正奖励，随距离、速度、姿态和双腿接触程度衰减
    w_soft = 1.0
    prox_thresh = 0.5
    vel_thresh = 0.5
    ang_thresh = 0.5

    proximity_gate = max(0.0, 1.0 - next_dist / prox_thresh)
    vel_mag = abs(next_vx) + abs(next_vy)
    vel_gate = max(0.0, 1.0 - vel_mag / vel_thresh)
    ang_gate = max(0.0, 1.0 - abs(body_angle) / ang_thresh)
    contact_ratio = (left_contact + right_contact) / 2.0

    soft_landing_bonus = w_soft * proximity_gate * vel_gate * ang_gate * contact_ratio

    # 3. 碰撞预防（保留 iter 2 结构，几乎无触发）
    x_bound = 0.9
    w_crash_x = 10.0
    penalty_crash_x = 0.0
    if abs(next_x) > x_bound:
        penalty_crash_x = -w_crash_x * (abs(next_x) - x_bound) ** 2

    danger_vy = -0.5
    w_crash_y = 5.0
    penalty_crash_y = 0.0
    if next_y < -0.1 and next_vy < danger_vy:
        penalty_crash_y = -w_crash_y * (next_vy - danger_vy) ** 2

    # 4. 时间惩罚（使用 iter 2 的轻量值）
    living_penalty = -0.002

    total_reward = (reward_prox + soft_landing_bonus +
                    penalty_crash_x + penalty_crash_y +
                    living_penalty)

    components = {
        "goal_proximity": reward_prox,
        "soft_landing_bonus": soft_landing_bonus,
        "crash_prevention_x": penalty_crash_x,
        "crash_prevention_y": penalty_crash_y,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```