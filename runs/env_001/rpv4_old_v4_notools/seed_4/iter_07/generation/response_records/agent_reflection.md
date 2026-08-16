# Response Record

# 1. 诊断报告

**evidence**  
- 上一轮 iter_6 的 score=-111.9，len=69.8，terminated 20/20，early_terminal 18/20（大部分在 150 步内得分低于 -50 就终止）。这表明飞行器几乎总是在早期就坠毁。  
- 奖励分量中：progress ep_sum_mean=55.7 (active 100%)，是唯一的大正奖励组件，但它定义为 `(dist_obs - dist_next) * 50`，鼓励“每一步减小距离”，而不关心速度。agent 学会的最直接策略就是失控下坠去获取这个 reward。  
- attitude 组件（-15）和 fuel 组件（-0.6）均为负，但力度不足以阻止下坠。  
- landing 组件 active_rate 仅 0.6%，且 mean -5.7，说明仅当偶尔触地时才激活，而且因为 `speed_factor` 在高速时变负，导致 land 变成惩罚，这进一步抑制了 agent 尝试着陆。  
- checkpoint 趋势显示 score 始终在 -115 ~ -106 间波动，无改善迹象，组件趋势同样平坦，没有学习改进的证据。

**behavior_diagnosis**  
策略已坍缩为“快速冲向原点（目标垫板位置）以收割 progress reward”，完全忽略速度与姿态控制，导致高速撞击地面/垫板，连续提前终止。这是一个典型的 reward hacking 行为。

**signal_completeness**  
缺少两个核心信号：  
1. 对速度的全局惩罚——迫使 agent 在飞行全程主动减速；  
2. 真正的“成功软着陆”一次性正奖励——目前 land 组件不仅不奖励成功，反而惩罚接触，彻底错位。  
此外，progress 奖励的“增量”形式（Δ-dist）天然鼓励快速接近而非安全接近，需要改变为主观“当前距离”评价。

**selected_level**：L3（主信号框架重写）  
理由：得分始终在 -100 附近，从不接近零或正，属于“score始终接近零或负”的情况，必须抛弃当前 progress‑driven 框架。

**selected_intervention**  
- 将 progress 替换为 **距离奖励**（针对当前状态的函数，而非增量），例如 `w_dist / (1 + dist_next)`，保证越近越好但不会奖励快速坠落。  
- 新增 **全局速度惩罚** `-w_vel * velocity_mag`，迫使 agent 全程控制速度。  
- 将 landing 改为 **一次性成功着陆大幅正奖励**，触发条件：双足接触、距离 < 阈值、速度 < 阈值。非触发时该分量为 0，不再有惩罚。  
- 保留 attitude 和 fuel 惩罚，系数基本不变。

**falsifiable_hypothesis**  
新 reward 结构使“快速坠落”不再有利（距离奖励有上限，而速度惩罚随速度增大），迫使 agent 学习减速并向中心接近；最终成功着陆的高额 bonus 成为唯一大规模正反馈来源，将策略收敛到期望的软着陆行为。

**expected_next_round**  
episode 长度显著增加，early_terminal 比例下降，agent 将学会在下降过程中主动使用主引擎减速，终末有部分 episode 能触发 land_bonus，score 会从 -100 跳变到明显更高的区间（理想情况下可出现正分），land 组件 active_rate 会显著上升（从 0.6% 提高到至少 10% 以上）。

---

# 2. 完整代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- extract signals ---
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- helpers ---
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    velocity_mag = (vx_next ** 2 + vy_next ** 2) ** 0.5

    # === 1. distance reward — closer is better, but capped ===
    w_dist = 10.0
    distance_reward = w_dist * (1.0 / (1.0 + dist_next))

    # === 2. global velocity penalty — force deceleration everywhere ===
    w_vel = 5.0
    velocity_penalty = -w_vel * velocity_mag

    # === 3. attitude stabilization (as before) ===
    w_angle = 1.0
    w_angvel = 0.5
    attitude_penalty = -w_angle * (angle_next ** 2) - w_angvel * (angvel_next ** 2)

    # === 4. fuel penalty (same as before but slightly scaled) ===
    if action == 0:
        fuel_cost = 0.0
    elif action == 2:   # main engine
        fuel_cost = 0.2
    else:               # side engines
        fuel_cost = 0.1
    fuel_penalty = -fuel_cost

    # === 5. soft landing BONUS — large positive, only on successful touchdown ===
    landing_bonus = 0.0
    if left_contact and right_contact:
        if dist_next < 0.2 and velocity_mag < 0.5:
            landing_bonus = 200.0   # one-shot completion reward

    # --- total ---
    total_reward = (distance_reward +
                    velocity_penalty +
                    attitude_penalty +
                    fuel_penalty +
                    landing_bonus)

    components = {
        "distance": distance_reward,
        "velocity": velocity_penalty,
        "attitude": attitude_penalty,
        "fuel": fuel_penalty,
        "landing": landing_bonus,
    }

    return float(total_reward), components
```
