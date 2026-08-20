`evidence`: 历史最佳(iter1)使用乘法结构得分2926、episode满1000步存活；iter3改为加法解耦后得分降至2510、terminated升至11/20、episode长度降至828；当前reward中speed_posture_reward占99% magnitude_share，ang_vel_penalty仅1%，缺少对动作效率的约束。上一轮尝试复制iter3加法结构被拒，需要实质性不同的干预。

`behavior_diagnosis`: 策略在iter1乘法结构下能存活完整回合但前进速度不足（得分2926 vs 目标3800），推测为保守运动模式——躯干保持直立但缺乏有力的跳跃推进；iter3加法解耦后稳定性恶化，频繁提前终止。

`signal_completeness`: 现有信号具备前进引导(forward_vel)和姿态稳定(posture_factor + ang_vel_penalty)，但缺乏动作效率或能耗约束。环境事实中action是可用信号且未被使用，属于可补充的职责。

`selected_level`: **Level 2**。触发条件：必要职责缺失（动作效率约束），且Level 1仅调系数已在iter1→iter2→iter3中反复尝试未刷新best。

`selected_intervention`: 在iter1乘法骨架基础上**新增energy_penalty组件**（对三关节扭矩的平方和施加轻量惩罚），不改变speed_posture_reward和ang_vel_penalty的现有数学形态。

`falsifiable_hypothesis`: 对关节扭矩施加轻量效率惩罚将抑制浪费性高频抖动，促使策略学习更协调的跳跃运动模式，从而在保持存活的前提下提高有效前进速度；若该假设成立，应观察到energy_penalty的active_rate达到100%且episode仍接近全长，同时score上升。

`expected_next_round`: episode_length应保持接近1000（不因新惩罚显著缩短），speed_posture_reward的episode_sum_mean应上升，energy_penalty的magnitude_share应在3-10%范围内（轻量但不为零），score应向target 3800方向改善。

`main_risk`: 能量惩罚可能进一步抑制探索性大扭矩动作，导致前进速度不升反降；若出现此模式，下一轮应移除该惩罚并尝试其他结构变换。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取关键运动信号
    torso_angle = next_obs[1]           # 躯干倾角
    forward_vel = next_obs[5]           # 前进速度（水平速度）
    torso_ang_vel = next_obs[7]         # 躯干角速度

    # 姿态因子：历史最佳iter1的指数衰减结构，躯干越直立因子越接近1
    k_angle = 10.0
    posture_factor = 2.718281828 ** (-k_angle * (torso_angle ** 2))

    # 主学习信号：前进速度 × 姿态因子
    #   前进时奖励，后退时惩罚；姿态倾斜时自动衰减，抑制为速度而过度倾身
    speed_posture_reward = forward_vel * posture_factor

    # 轻量稳定约束：惩罚躯干角速度，抑制快速晃动/摔倒
    w_ang_vel = 0.05
    ang_vel_penalty = w_ang_vel * abs(torso_ang_vel)

    # 动作效率惩罚：对三关节扭矩平方和施加轻量约束
    #   抑制高频抖动和浪费性大力矩，引导协调高效的跳跃模式
    #   系数0.05：动作平方和在[0,3]范围，单步罚约0~0.15，约为speed_posture的3%~10%
    w_energy = 0.05
    energy_penalty = w_energy * (action[0]**2 + action[1]**2 + action[2]**2)

    total_reward = speed_posture_reward - ang_vel_penalty - energy_penalty

    components = {
        'speed_posture_reward': speed_posture_reward,
        'ang_vel_penalty': ang_vel_penalty,
        'energy_penalty': energy_penalty
    }

    return float(total_reward), components
```