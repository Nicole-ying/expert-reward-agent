# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取当前和下一时刻的位置
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    # 速度、姿态、接触信息（使用 next_obs 更合理，反映动作导致的后果）
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 主学习信号：进度增量奖励 ----------
    # 每一步奖励“当前到目标的距离”与“下一步到目标的距离”之差
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5
    progress_delta = dist - next_dist   # 正值表示更靠近目标

    # ---------- 稳定/安全惩罚 ----------
    # 惩罚水平、垂直速度，以及姿态角和角速度，鼓励稳定接近
    stability_penalty = -(
        0.1 * abs(vx) +
        0.1 * abs(vy) +
        0.2 * abs(angle) +
        0.1 * abs(ang_vel)
    )

    # ---------- 任务完成近似信号（软着陆 proxy） ----------
    # 同时满足：靠近中心、低速、姿态稳定、双支撑脚接触，则给予小奖励
    dist_thresh = 0.5
    vel_thresh = 0.2
    angle_thresh = 0.1
    ang_vel_thresh = 0.1

    if (next_dist < dist_thresh and
        abs(vx) < vel_thresh and
        abs(vy) < vel_thresh and
        abs(angle) < angle_thresh and
        abs(ang_vel) < ang_vel_thresh and
        left_contact > 0.5 and right_contact > 0.5):
        soft_landing_proxy = 1.0
    else:
        soft_landing_proxy = 0.0

    # ---------- 总奖励 ----------
    w_progress = 5.0
    w_stab = 1.0      # stability_penalty 内部已含负号，直接加
    w_soft = 1.0

    total_reward = (
        w_progress * progress_delta +
        w_stab * stability_penalty +
        w_soft * soft_landing_proxy
    )

    components = {
        "progress_delta_reward": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及其角色

1. **progress_delta_reward（主学习信号）**  
   计算 `d(obs, goal) - d(next_obs, goal)`，直接度量每一步向目标（中心着陆垫）的进度。该分量提供密集、连续的学习引导，使智能体快速学会朝向目标移动。

2. **stability_penalty（稳定/安全约束）**  
   惩罚水平速度、垂直速度、主体朝向角和角速度，鼓励飞行器在接近过程中保持平稳。权重较轻，不会压制探索，但能抑制高速撞击和剧烈旋转，为后续精细降落打下基础。

3. **soft_landing_proxy（任务完成近似信号）**  
   当同时满足“靠近着陆垫中心”、“低速”、“角度稳定”、“双支撑脚接触”时，给予一个小的正向奖励。这是一个 soft proxy，不能替代 success flag，但能为成功停留提供额外激励。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- **explicit_success_flag_available = false**，info 字段为空，无法可靠判断“成功完成”事件。若强行添加 terminal_success_reward，将导致 agent 借助未声明的信号，造成虚假奖励。
- **explicit_failure_flag_available = false**，同样无法区分坠毁、
