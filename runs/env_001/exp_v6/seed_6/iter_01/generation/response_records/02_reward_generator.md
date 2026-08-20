# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract states ----------
    # obs
    x, y = obs[0], obs[1]
    # next_obs
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 1. main learning signal: progress toward (0,0) ----------
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress_delta_reward = dist_obs - dist_next

    # ---------- 2. stability / smoothness penalty ----------
    # lightweight penalties to encourage gentle, stable landing
    vel_penalty = abs(vx) + abs(vy)          # prefer zero horizontal/vertical speed
    angle_penalty = abs(angle)               # prefer upright orientation
    ang_vel_penalty = abs(ang_vel)           # prefer no rotation

    w_vel = 0.01
    w_angle = 0.05
    w_angvel = 0.01
    stability_penalty = - (w_vel * vel_penalty + w_angle * angle_penalty + w_angvel * ang_vel_penalty)

    # ---------- 3. soft landing proxy (small bonus for likely safe touchdown) ----------
    # Criteria: near target, low speed, upright, both legs in contact
    near_target = dist_next < 0.2
    low_speed = abs(vx) < 0.1 and abs(vy) < 0.1
    upright = abs(angle) < 0.05
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)

    soft_landing_proxy = 0.1 if (near_target and low_speed and upright and both_contact) else 0.0

    # ---------- total ----------
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy

    # ---------- components dict ----------
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及角色

1. **progress_delta_reward** – 主学习引导  
   计算当前步接近目标 (0,0) 的距离变化。正值鼓励每一步缩短与目标的欧氏距离，提供密集的到达信号，是本次导航-目标到达任务的核心驱动。

2. **stability_penalty** – 稳定/平滑约束  
   小权重惩罚水平与垂直速度的绝对值、机身姿态倾角以及角速度。作用相当于“软着陆倾向”，引导飞行器以低速、竖直姿态接近目标，避免高速撞击或剧烈翻滚。

3. **soft_landing_proxy** – 任务完成近似信号  
   当飞行器同时满足：靠近目标、低速度、接近竖直、双脚接触时，给予一个小的正奖励。用于标识极可能已经安全着陆的瞬态，辅助收敛到真正的终端状态。权重极小（0.1），防止替代真实成功信号。

## 未使用 terminal_success_reward / terminal_failure_penalty 的原因

环境 `info` 字典为空，不存在 `success` 或 `failure` 等显式标志，根本无法可靠地判断终结态是成功还是失败。强行使用会诱发模型“发明”不存在的信息，因此这两项不纳入 v1 核心。

## 留到后续迭代的组件

- **terminal_success_reward / terminal_failure_penalty**：一旦 wrapper 或者观测提供可靠的终止原因标志（如 `info['success']`）时，可加入以加速末端学习。
- **energy_penalty（动作成本）**：本次未加入，避免早期“害怕移动”。待飞行器能稳定接近目标后，可增加对主引擎/姿态引擎使用的轻微惩罚，优化燃料消耗。
- **time_penalty**：后续如果出现接近目标却长时间徘徊不降落的情况，可引入极小的每步时间惩罚。
- **gated_reward 或更严格的 shaping**：若发现 `progress_delta_reward` 在某些场景下引发目标附近震荡，可在后续用 clipping 或基于势能的 PBRS 替代。

## 训练后需观察的 failure mode

- **high_reward_without_success**：即使奖励很高，飞行器也可能在目标上方来回摆动而不稳定着陆。需检查 reward 曲线是否与真实成功率解耦。
- **fast_crash_near_goal**：`progress_delta` 可能使飞行器快速冲向目标，导致高速撞击提前终止。需要监控速度和姿态统计，必要时增大稳定性惩罚权重。
- **contact reward hacking**：`soft_landing_proxy` 若条件过宽（例如单腿触碰边缘），可能被滥用。应通过环境接触次数与最终奖励的分布监测，若出现异常，收紧 `near_target`/`low_speed` 的阈值或移除该项。
