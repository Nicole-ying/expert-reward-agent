1. evidence：首轮评估全部20回合以terminated快速失败(平均68.75步)，得分-92.75；生成的奖励组件总和仅约2.49，正信号太弱，无法抵消环境每步约-1.42的基础惩罚；stable_bonus仅16.9%触发，提供稀疏但份额较大的奖励，说明精细停靠指导严重不足。
2. behavior_diagnosis：agent 在很短步数内坠毁或出界，未能靠近平台中心，缺乏足够的靠近引导使其无法克服环境负步数惩罚，导致所有回合均以失败终止。
3. signal_completeness：缺少密集的“距离目标”正反馈，当前仅依靠较小的progress增量和极稀疏的停靠bonus，导致正向信号过弱；失败惩罚缺失但本轮不引入；未使用的角速度(obs[5])暂时不构成缺口主因。
4. selected_level：Level 2 — 结构变换：增加一个新的`approach_reward`组件，将二值/稀疏的接近信号转化为连续bounded factor，弥补密集接近反馈的缺失。
5. selected_intervention：新增组件`approach_reward = 0.1 / (1.0 + distance)`，利用已使用的位置观测生成针对距离的密集正奖励，保留原有三个组件不变。
6. falsifiable_hypothesis：密集距离奖励将通过每步直接给予靠近目标的正面信号，使agent更倾向于飞向平台中心而不是漂移坠毁，从而延长存活步数、减少快速失败，并提高总得分（负得分收窄或变正）。
7. expected_next_round：下一轮评估中`approach_reward`的active_rate应≈100%，magnitude_share成为主要奖励来源；episode_length应显著增加（如>100步），score应提高至零附近或更高；`goal_progress`和`stable_bonus`的绝对值可能因更长存活而有所上升，但仍为次要信号。
8. main_risk：仅奖励距离可能鼓励agent不顾速度直接冲向中心，导致高速撞击而失败；需要后续结合速度或姿态约束防止该漏洞，但本轮先建立基本引导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 航向进展：距离目标越近越好（improvement_delta）
    d_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = d_prev - d_next
    goal_progress = 1.0 * progress

    # 2. 稳定停靠奖励：靠近目标时鼓励低速、竖直、双腿接触
    proximity_thresh = 0.5
    proximity_gate = max(0.0, 1.0 - d_next / proximity_thresh)

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    vel_thresh = 0.2
    velocity_bonus = 0.5 * max(0.0, 1.0 - speed / vel_thresh)

    angle_thresh = 0.1
    angle_bonus = 0.2 * max(0.0, 1.0 - abs(next_obs[4]) / angle_thresh)

    contact_bonus = 1.0 * next_obs[6] * next_obs[7]

    stable_bonus = proximity_gate * (velocity_bonus + angle_bonus + contact_bonus)

    # 3. 燃料效率惩罚
    fuel_penalty = -0.01 if action != 0 else 0.0

    # 4. 密集距离奖励：越接近目标奖励越大（连续有界）
    approach_reward = 0.1 / (1.0 + d_next)

    total_reward = goal_progress + stable_bonus + fuel_penalty + approach_reward
    components = {
        'goal_progress': float(goal_progress),
        'stable_bonus': float(stable_bonus),
        'fuel_penalty': float(fuel_penalty),
        'approach_reward': float(approach_reward)
    }
    return float(total_reward), components
```