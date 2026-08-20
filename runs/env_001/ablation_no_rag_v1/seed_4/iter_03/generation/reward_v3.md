1. `evidence`: 上一轮得分-87.67、best -53.29，当前episode满1000步且target_proximity magnitude_share 100%，正奖励使策略徘徊而忽略着陆；best长度72.7且target_proximity为线性负惩罚，但同样缺乏着陆信号。
2. `behavior_diagnosis`: 当前策略学会在目标附近徘徊以持续获取target_proximity正奖励，未完成着陆，外部得分差；best策略可能因缺乏着陆引导而快速失败。
3. `signal_completeness`: 必要职责缺失：无着陆或任务完成信号，仅有位置接近引导，导致proxy徘徊或过早终止。
4. `selected_level`: Level 2，触发条件为缺少任务完成信号且proxy引导导致策略不完成着陆（proxy_to_completion_alignment）。
5. `selected_intervention`: 以best代码为基础，添加landing_reward组件：当左右支撑接触标志均为1时给予+1奖励，系数1.0。
6. `falsifiable_hypothesis`: 加入着陆奖励后，策略将学习尽快着陆并保持接触，从而触发成功终止，外部得分提升，landing_reward激活率上升。
7. `expected_next_round`: episode长度从1000缩短，外部得分提高，landing_reward激活率大于0，target_proximity magnitude_share下降。
8. `main_risk`: 着陆奖励可能导致过早尝试着陆而坠毁，或着陆后不稳定导致反复离开平台，但结合已存在的速度与姿态惩罚可抑制不稳定行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]          # 相对目标平台水平坐标
    y = next_obs[1]          # 相对平台高度
    vx = next_obs[2]         # 水平速度
    vy = next_obs[3]         # 垂直速度
    angle = next_obs[4]      # 机身角度
    omega = next_obs[5]      # 角速度
    left_contact = next_obs[6]   # 左支撑接触标志
    right_contact = next_obs[7]  # 右支撑接触标志

    w_x = 0.2
    w_y = 0.2
    w_vx = 0.05
    w_vy = 0.05
    w_angle = 0.05
    w_omega = 0.05
    w_landing = 1.0

    target_proximity = -w_x * abs(x) - w_y * abs(y)
    velocity_penalty = -w_vx * (vx ** 2) - w_vy * (vy ** 2)
    orientation_penalty = -w_angle * (angle ** 2)
    angvel_penalty = -w_omega * (omega ** 2)
    landing_reward = w_landing * (left_contact * right_contact)  # 只有双支撑接触时为正

    total_reward = target_proximity + velocity_penalty + orientation_penalty + angvel_penalty + landing_reward

    components = {
        "target_proximity": target_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_reward": landing_reward
    }

    return float(total_reward), components
```