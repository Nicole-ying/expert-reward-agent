# Response Record

根据训练反馈分析：

1. **Agent 发生了什么？**  
   - 上一轮（iter5）加入了 `alive_penalty=-0.005` 和 `main_engine_penalty=-0.02`，同时将 `distance_progress` 系数提高到 5.0。  
   - 导致所有评估 episode 的 `contact_reward` 和 `landing_bonus` 激活率均为 0，说明 agent **从未让任何支撑腿接触地面**，完全没有学会着陆行为。  
   - 最终 reward 被主引擎惩罚（-46%）和存活惩罚（-20%）主导，`distance_progress` 虽为正但很小（+13% 份额），agent 只是勉强向中心漂移，无法触发着陆。  
   - `episode_length` 长达 876，部分 episode 被截断，部分可能坠毁/失败终止，整体行为严重退化，score 跌至 -119.3。  

2. **最值得干预的组件**：  
   - 新增的 `alive_penalty` 和 `main_engine_penalty` 破坏了之前成功的学习。在 iter4（best score=130.6）中，**没有生存惩罚和燃料惩罚**，agent 成功学会了使用主引擎并完成软着陆，获得了可观的 `landing_bonus`。  
   - 数学形态上，这两个连续负奖励分量叠加后压倒了稀疏的着陆正向信号，导致 agent 避开主引擎、放弃下降，陷入局部悬停策略。  
   - 因此最值得干预的是**移除这两个破坏性惩罚，恢复 best 的主体奖励结构**。  

3. **上一轮修改回顾**：  
   - 添加了 `alive_penalty` 和 `fuel_penalty`，增大了 `distance_progress` 系数。  
   - 实际效果：严重恶化，score 从 130.6 暴跌至 -119.3。  

**修改方案**：  
- 以 best（iter4）结构为基础，保留 `angle_penalty`、`angvel_penalty`、`contact_reward`、`distance_progress`、`landing_bonus` 五大组件。  
- 完全移除 `alive_penalty` 和 `main_engine_penalty`。  
- 将 `distance_progress` 系数从 5.0 降回更温和的 2.0，避免过强引导与稀疏着陆奖励产生冲突，同时仍提供足够的趋向目标梯度。  
- 保留原有软着陆乘法奖励形式（阈值 0.3/0.2/0.3），不做增加新假设的修改。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取状态
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # 距离进展
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    progress = prev_dist - next_dist
    c_progress = 2.0 * progress

    # 腿部接触奖励
    contact_count = n_left_contact + n_right_contact
    c_contact = 2.0 * contact_count

    # 姿态惩罚
    c_angle = -0.05 * (n_angle**2)
    c_angvel = -0.01 * (n_angvel**2)

    # 软着陆奖励（仅双腿同时接触）
    two_legs = (n_left_contact > 0.5 and n_right_contact > 0.5)
    if two_legs:
        dist = next_dist
        speed_mag = (nvx**2 + nvy**2) ** 0.5
        angle_mag = abs(n_angle)
        angvel_mag = abs(n_angvel)

        score_dist = max(0.0, 1.0 - dist / 0.3)
        score_speed = max(0.0, 1.0 - speed_mag / 0.3)
        score_angle = max(0.0, 1.0 - angle_mag / 0.2)
        score_angvel = max(0.0, 1.0 - angvel_mag / 0.3)

        c_landing = 200.0 * score_dist * score_speed * score_angle * score_angvel
    else:
        c_landing = 0.0

    total_reward = c_progress + c_contact + c_angle + c_angvel + c_landing

    components = {
        'distance_progress': c_progress,
        'contact_reward': c_contact,
        'angle_penalty': c_angle,
        'angvel_penalty': c_angvel,
        'landing_bonus': c_landing,
    }
    return (float(total_reward), components)
```
