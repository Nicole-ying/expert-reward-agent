# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # parameters
    w_align = 0.8
    w_vel = 0.01
    w_angle = 0.01
    w_angvel = 0.005
    landing_bonus = 0.5

    # velocity alignment reward (main learning signal)
    # encourage velocity directed toward the goal at (0,0)
    # dot product: -(x * vx + y * vy) -> positive when moving toward origin
    dot = -(next_obs[0] * next_obs[2] + next_obs[1] * next_obs[3])
    # clamp to [-1,1] to avoid huge values, then only positive part
    dot_clamped = max(min(dot, 1.0), -1.0)
    alignment_reward = w_align * max(0.0, dot_clamped)

    # stability penalty (quadratic form, lighter on small deviations)
    vx, vy = next_obs[2], next_obs[3]
    vel_penalty = w_vel * (vx**2 + vy**2)
    angle_penalty = w_angle * (next_obs[4]**2)
    angvel_penalty = w_angvel * (next_obs[5]**2)
    stability_penalty = -(vel_penalty + angle_penalty + angvel_penalty)

    # soft landing proxy bonus (multi‑condition completion signal)
    dist_to_goal = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    speed_mag = (vx**2 + vy**2) ** 0.5
    near_target = dist_to_goal < 0.3
    low_speed = speed_mag < 0.2
    stable_angle = abs(next_obs[4]) < 0.05
    both_contact = (next_obs[6] == 1.0 and next_obs[7] == 1.0)
    soft_landing_proxy = landing_bonus if (near_target and low_speed and stable_angle and both_contact) else 0.0

    total_reward = alignment_reward + stability_penalty + soft_landing_proxy

    components = {
        "velocity_alignment_reward": alignment_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **velocity_alignment_reward**（主学习信号）：  
  奖励智能体当前速度方向指向目标点（0,0）。计算 `-(x*vx + y*vy)`，只取正值并限制在 [-1,1] 范围内。该信号稠密，每步都能提供梯度，但不同于基于距离变化量或绝对距离的奖励——它直接引导智能体朝目标移动，即使距离还很远也有明确的方向性奖励。避免在目标附近因距离梯度消失而震荡；当越过分目标时点积变负（奖励消失），促使智能体回正。

- **stability_penalty**（稳定约束）：  
  使用二次惩罚抑制过大的线速度、机体角度和角速度。二次形式对小偏差容忍度高，对大偏差惩罚显著，避免之前线性惩罚过高压制主信号。权重设置较小，主要防止高速翻滚，不抑制正常移动。

- **soft_landing_proxy**（任务完成近似信号）：  
  多条件组合奖励：必须同时满足距离中心 <0.3、整体速度 <0.2、角度 <0.05 且两侧支撑腿同时接触。只在所有条件成立时给予一个小常数奖励（0.5），作为着陆成功的近似。由于没有显式 success flag，该组件用于桥接“靠近目标”与“稳定停靠”，其触发条件严苛，避免被单一接触信号利用。

- **为何不用 terminal_success_reward / terminal_failure_penalty**：  
  环境卡片明确声明 `explicit_success_flag_available=false` 且 info 字段为空，不存在可靠的终止状态标志，因此避免使用任何基于成功/失败的稀疏终端奖励。

- **后续迭代预留**：  
  当智能体能稳定靠近并偶尔触发 soft_landing_proxy 后，可考虑加入轻微的动作效率惩罚（energy_penalty）以鼓励减少引擎使用；若出现高速接近但无法稳定着陆的现象，可收紧 soft_landing_proxy 的条件或引入更复杂的门控动态调整。

- **需观察的 failure mode**：  
  - **alignment 只引导方向，不保证减速**：若 stability_penalty 过弱，智能体可能高速冲向目标然后飞出，需监控速度指标。  
  - **soft_landing_proxy 触发率极低**：初始阶段可能几乎为零，导致该奖励形同虚设，此时可考虑稍放宽条件（如增大距离阈值）或增大步中引导信号。  
  - **在目标附近无奖励时原地振荡**：由于 velocity_alignment 在目标附近幅度很小，若缺乏其他信号，智能体可能失去动力。此时 stability_penalty 应帮助其静止，但若 weight 太小可能仍会乱动，可通过观察角速度和位置方差来判断。
