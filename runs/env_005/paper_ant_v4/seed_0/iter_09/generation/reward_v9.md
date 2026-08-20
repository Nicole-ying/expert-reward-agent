1. `evidence`：当前策略在多数episode中提前终止（13/20），但forward_reward均值极高（1523），伴有大姿态和角速度惩罚，说明agent为高速前冲而牺牲稳定性，高度惩罚极低（-34）且主动率仅12.1%，无法阻止摔倒。
2. `behavior_diagnosis`：agent学会了高速奔跑，但姿态控制很差，频繁摔倒，导致episode提前结束；存活episode中累积了高前进奖励，但整体得分被巨大惩罚抵消，且高度约束过弱是摔倒的重要推手。
3. `signal_completeness`：前进、姿态、横向和角速度约束已存在，但高度保护职责过于软，缺失能在接近摔倒时产生强回馈的组件；加入强烈的高度边缘惩罚可弥补这一缺失。
4. `selected_level`：Level 2，因为同一骨架已连续两轮未刷新best（iter 7→8下降），且目标完成率仅7.4%；高度惩罚尺度太弱，改变其数学形态以形成有效保护。
5. `selected_intervention`：唯一目标组件为`height_penalty`，将低高度惩罚由线性改为二次且阈值从0.3提前到0.4，系数从20提升到100，使其在接近摔倒时急剧增大。
6. `falsifiable_hypothesis`：加强低高度惩罚将迫使agent在维持更高身体高度的前提下探索前进，减少摔倒，从而延长有效episode并提升总得分，而不会完全阻断前进学习。
7. `expected_next_round`：`height_penalty`的active_rate和magnitude_share上升；`terminated`比例下降；`episode_length`增加；`score`提高，至少走向之前的best。
8. `main_risk`：惩罚过强可能使得agent过于保守，停在原地不敢前进；若下一轮得分停滞或进一步下降，需平衡前进奖励强度或引入存活小奖励。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取状态
    body_z = next_obs[0]
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]

    # 前进奖励：平方强化高速
    fwd_speed = max(0.0, body_x_vel)
    forward_reward = 15.0 * (fwd_speed ** 2)

    # 强化的高度惩罚：低阈值提前到0.4，并用二次放大靠近摔倒的代价
    height_low = max(0.0, 0.4 - body_z)
    height_high = max(0.0, body_z - 0.9)
    height_penalty = 100.0 * (height_low ** 2 + height_high)

    # 直立惩罚：连续惩罚，偏离完全直立即产生代价
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_penalty = 5.0 * ((1.0 - body_up_z) ** 2)

    # 横向速度惩罚
    lat_penalty = 2.0 * (body_y_vel ** 2)

    # 角速度惩罚
    ang_penalty = 0.5 * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    total_reward = forward_reward - height_penalty - upright_penalty - lat_penalty - ang_penalty

    components = {
        'forward_reward': forward_reward,
        'height_penalty': -height_penalty,
        'upright_penalty': -upright_penalty,
        'lat_penalty': -lat_penalty,
        'ang_penalty': -ang_penalty
    }
    return float(total_reward), components
```