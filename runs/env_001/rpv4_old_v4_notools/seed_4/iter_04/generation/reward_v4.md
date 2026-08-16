**1. 诊断报告**

- `evidence`：  
  - 本轮 score = -110.1，全部 20 集均为 early terminal（<150 步且 score<-50），且 terminated=20/20（非截断，即因失败终止）。  
  - landing_contact active_rate 仅 0.5%（极度稀疏），ep_sum_mean=3.5 说明极少数 episode 在失败边缘触发了着陆奖励，但对整体无引导力。  
  - goal_proximity 全程稳定在 ~10.86，无明显上升趋势；speed_penalty 平坦，attitude_constraint 微有改善但仍负。  
  - 由于环境会给坠毁/出界等 terminal failure 一个较大的负惩罚（未在 component 统计中体现），导致 score 始终大幅为负。  
  - checkpoint 趋势中，score 从 -118.6 到 -110.1，改善极微，说明当前奖励未能教会 agent 安全着陆。

- `behavior_diagnosis`：  
  当前策略在目标接近 shaping 的驱动下，会快速向中心移动，但缺乏着陆所需的减速与接触引导。接近时高速撞击目标区域，触发了环境的坠毁惩罚，导致 episode 提前终止，分数极低。奖励中虽有速度惩罚，但力度不足以产生安全着陆速度；landing_contact 作为稀疏事件，只有最终完美条件才给奖励，无法提供学习梯度。

- `signal_completeness`：  
  缺少一个**连续的着陆过程奖励**。现有信号已涵盖靠近（goal_proximity）、降速（speed_penalty）、姿态（attitude_constraint），但着陆的最终阶段——在目标区域以低速、正确接触地面——没有任何中间奖励。这导致 agent 只能靠稀疏的 landing_contact 盲目探索，几乎学不会。

- `selected_level`：**L2**  
  理由：增减系数（L1）无法解决着陆奖励稀疏的本质问题；必须修改 landing_contact 组件的数学形态，将其从离散终态事件变为连续函数。

- `selected_intervention`：  
  将 landing_contact 组件从 `if near_center and low_speed and both_contacts: landing_contact = 10.0` 改为**基于高斯位置接近度、指数速度衰减与接触分数的乘积的连续奖励**。新公式：
  ```
  nearness = exp(-dist² / (2·0.15²))
  slowness = exp(-speed_norm / 0.3)
  contact_score = (left_contact + right_contact) / 2
  landing_contact = 3.0 * nearness * slowness * contact_score
  ```
  该奖励在接近中心、低速、双腿接触时给予正向连续信号，无界→有界，避免了极端值支配，同时保留了稀疏完美接触的引导潜力。

- `falsifiable_hypothesis`：  
  若连续着陆奖励能提供足够的梯度，agent 将学会在目标区域减速并尝试接触地面；early termination 率将明显下降，部分 episode 可能实现安全接触，score 应从 -110 逐步升至 -50 或更高。如果 active_rate 仍极低且 episode 依然全部 early terminate，则说明连续奖励形式或系数仍需调整，但至少打破了稀疏无梯度的困境。

- `expected_next_round`：  
  agent 仍会先快速接近目标，但在接近阶段（dist<0.5）会产生较大的 nearness 与 slowness 奖励，促使减速。预期 landing_contact 的 active_rate 将从 0.5% 升至 20%–40%，episode_sum 上升，部分 episode 可能实现软着陆，score 负值减小，可能达到 -40 ~ -20 区间。

---

**2. 完整 Python 代码**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v5 — 着陆引导连续化。
    保持目标接近势能差、距离门控速度惩罚，
    将 landing_contact 从稀疏事件改为位置/速度/接触的连续乘积奖励。
    """
    # 当前状态
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]

    # 下一状态
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # ========== A. 目标接近 shaping（势能差） ==========
    current_potential = -(abs(x_pos) + abs(y_pos))
    next_potential = -(abs(next_x_pos) + abs(next_y_pos))
    goal_proximity = 10.0 * (next_potential - current_potential)

    # ========== B. 速度惩罚（距离门控版） ==========
    current_speed = abs(x_vel) + abs(y_vel)
    next_speed = abs(next_x_vel) + abs(next_y_vel)
    avg_speed = (current_speed + next_speed) / 2.0
    dist_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    gating = 1.0 / (1.0 + 5.0 * dist_to_target)
    speed_penalty = -0.2 * avg_speed * gating

    # ========== C. 姿态约束 ==========
    angle_penalty = -0.5 * abs(next_body_angle)
    ang_vel_penalty = -0.2 * abs(next_ang_vel)
    attitude_constraint = angle_penalty + ang_vel_penalty

    # ========== D. 安全着陆接触奖励（连续版） ==========
    # 到中心的欧氏距离
    dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    # 位置接近度：高斯核，标准差0.15
    nearness = 2.718281828 ** (-dist ** 2 / (2 * 0.15 ** 2))
    # 速度低度：速度模（L1）的指数衰减，时间常数0.3
    speed_norm = abs(next_x_vel) + abs(next_y_vel)
    slowness = 2.718281828 ** (-speed_norm / 0.3)
    # 接触分数：双腿接触为佳
    contact_score = (next_left_contact + next_right_contact) / 2.0  # ∈ [0,1]
    # 连续着陆奖励，最大约 3.0
    landing_contact = 3.0 * nearness * slowness * contact_score

    # ========== E. 燃料效率 ==========
    fuel_efficiency = -0.05 * float(action != 0)

    # ========== 汇总 ==========
    total_reward = (
        goal_proximity +
        speed_penalty +
        attitude_constraint +
        landing_contact +
        fuel_efficiency
    )

    components = {
        'goal_proximity': goal_proximity,
        'speed_penalty': speed_penalty,
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency
    }

    return float(total_reward), components
```