根据诊断，原有 `progress_reward` 的线性差分形式允许负值，导致探索时巨大惩罚、提前终止。修改为 **仅正向奖励**（`max(0, dist_now - dist_next)`），消除坠落阶段的负反馈。同时修复函数签名和返回值格式，确保正确返回 `(float(total_reward), components)`。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Hyperparameters
    w_progress = 2.0          # 靠近目标的权重（仅正向）
    w_x = 0.2                 # 横向偏离惩罚权重
    w_engine = 0.05           # 引擎点火惩罚权重
    landing_bonus = 20.0      # 成功着陆一次性奖励

    # 着陆判断阈值
    contact_thresh = 0.5
    vx_landing_max = 0.5
    vy_landing_max = 0.5
    angle_landing_max = 0.1

    # 计算与目标的距离
    dist_now = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 进度奖励：只奖靠近，不罚远离
    progress_reward = w_progress * max(0.0, dist_now - dist_next)

    # 着陆奖励：双腿触地且速度/角度都在安全范围内
    left_contact = next_obs[6] > contact_thresh
    right_contact = next_obs[7] > contact_thresh
    low_speed_vx = abs(next_obs[2]) < vx_landing_max
    low_speed_vy = abs(next_obs[3]) < vy_landing_max
    small_angle = abs(next_obs[4]) < angle_landing_max

    if left_contact and right_contact and low_speed_vx and low_speed_vy and small_angle:
        landing_reward = landing_bonus
    else:
        landing_reward = 0.0

    # 横向位置惩罚
    lateral_penalty = -w_x * abs(next_obs[0])

    # 引擎使用惩罚
    engine_penalty = -w_engine * (1.0 if action != 0 else 0.0)

    # 总奖励
    total_reward = progress_reward + landing_reward + lateral_penalty + engine_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
        "lateral_penalty": lateral_penalty,
        "engine_penalty": engine_penalty
    }
    return float(total_reward), components
```