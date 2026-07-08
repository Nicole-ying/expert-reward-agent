# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward —— 每一步靠近目标的正向奖励
    dist_current = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_current - dist_next  # 正值表示靠近

    # 稳定与安全约束：对高速、大角度、快角速度的轻量惩罚
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    # 各系数较小，避免抑制正常机动
    stability_penalty = -0.05 * (abs(vel_x) + abs(vel_y)) - 0.1 * abs(angle) - 0.01 * abs(angular_vel)

    # 软着陆 proxy：当接近目标、低速、姿态小且双腿接触时给予一次性小奖励
    landing_bonus = 0.0
    left_contact = next_obs[6] > 0.5
    right_contact = next_obs[7] > 0.5
    if (dist_next < 0.1 and
        (vel_x**2 + vel_y**2) ** 0.5 < 0.2 and
        abs(angle) < 0.2 and
        left_contact and right_contact):
        landing_bonus = 1.0

    total_reward = progress_delta + stability_penalty + landing_bonus

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

---

# reward_v1 设计说明

## 使用的奖励组件与角色

1. **progress_delta_reward**（主学习信号）  
   - 角色：驱动智能体持续向目标着陆垫靠近。  
   - 数学形式：当前步到目标距离减去下一步到目标距离，正值越多表示进步越大。  
   - 信号来源：`obs[0], obs[1]` 和 `next_obs[0], next_obs[1]`。

2. **stability_penalty**（轻量稳定/安全约束）  
   - 角色：抑制过度速度、大姿态角和高速旋转，引导平滑着陆。  
   - 数学形式：对 `next_obs` 的速度、姿态角和角速度取绝对值，乘以小权重后作为负项。  
   - 权重设置：速度 0.05、角度 0.1、角速度 0.01，避免抑制正常机动。

3. **soft_landing_proxy**（任务完成近似信号）  
   - 角色：当智能体满足所有着陆条件（接近目标、低速、小姿态、双腿触地）时，给予一个小正向奖励，作为“成功着陆”的密集近似。  
   - 条件严格：位置距离 < 0.1，速率 < 0.2，姿态角 < 0.2 rad，双支撑腿均接触。奖励固定为 +1.0。  
   - 不会在没有同时满足所有条件时激活，防止 contact reward hacking。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境 **explicit_success_flag_available=false**，**explicit_failure_flag_available=false**，info 字典为空。  
- 无法可靠判断真正成功或失败终止，因此不能把它们作为核心项。本 v1 改用 soft proxy 来近似成功状态。

## 后续迭代预留的组件

- **energy_penalty / time_penalty**：v1 未加入，以免导致 agent 不敢动作或急于失败。当 v1 能稳定靠近目标后，再以小权重加入动作/时间效率项。
- **gated_reward / dynamic curriculum**：在多阶段训练或安全门控需求出现后再引入。
- **terminal_success / terminal_failure**：当 wrapper 能准确提供 success/failure 标志后，再替换或补充本 v1 的 soft landing proxy。

## 训练后应重点观察的 failure mode

- **high_reward_without_success**：智能体靠近目标并保持 low speed 但未能同时满足双腿接触，使得 progress_delta 变小但总奖励仍高。需检查是否 stuck 在目标附近悬停而不降落。
- **contact reward hacking**：若 soft landing proxy 条件过松，智能体可能单纯追求接触而忽略位置/速度，训练日志中 landing_bonus 频率异常高时应收紧阈值。
- **stability_penalty 过强**：若智能体不敢加速或长期静止，可适当降低惩罚系数。
- **震荡或 overshoot**：progress_delta 可能会在目标附近造成来回摆动，后续可考虑对 progress_delta 进行 clip 或加入阻尼项。