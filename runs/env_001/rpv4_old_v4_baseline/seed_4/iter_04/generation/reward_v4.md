我已经完整了解了整个演化历史。现在进行系统性诊断：

---

## 逐组件诊断

### 1. landing_contact — **核心问题，必须修复**

| 指标 | v1 | v2 (per-step, 2.0/leg) | v3 (transition, 25.0/leg) |
|---|---|---|---|
| ep_sum | 0.5 | 285.6 | **3090.0** |
| magnitude_share | 6.2% | 54.0% | **80.0%** |
| active_rate | 0.2% | 21.9% | 11.9% |

v3 的 transition-based 修改是灾难性的：ep_sum 暴涨 10.8 倍。原因是 **25.0/腿的一次性奖励太大**，agent 学会了在垫板上弹跳（contact 反复 0→1→0→1），每次 bounce 收割 25 分。v3 checkpoint 趋势中 landing_contact 从 cp10%=6.25 一路飙升到 cp100%=3037.5——这是教科书级的 exploit。

**但 v2 的 per-step 机制方向是对的**（score 178.7），只是 2.0/leg 的量级偏高导致 54% 占比。

**方案**：回归 v2 的 per-step 机制，但将系数从 2.0 降到 **0.5/leg**（L1）。预期 ep_sum 从 285.6 降至 ~71，占比从 54% 降至 ~15-20%。

### 2. goal_proximity — **健康，保留**

ep_sum 从 v1 的 2.1（死组件）→ v2 的 193（36.5%）→ v3 的 675（17.5%）。结构合理，指数邻近奖励提供了良好梯度。v3 中 17.5% 的占比被 landing_contact 的 80% 碾压，但这不是 goal_proximity 的问题。**保持不变**。

### 3. settling_stability — **被淹没，需要微调加强**

ep_sum: v2=-23.3 (4.4%) → v3=-46.4 (1.2%)。占比从已经偏低的 4.4% 进一步被 landing_contact 稀释到 1.2%。agent 几乎听不到速度惩罚信号。

**方案**：速度惩罚系数从 -0.3 → **-0.4**（L1）。预期占比从 1.2% → ~6-8%。

### 4. attitude_constraint — **同理被淹没**

ep_sum: v2=-10.7 (2.0%) → v3=-21.3 (0.6%)。角度误差系数 0.3 偏弱。

**方案**：角度误差权重从 0.3 → **0.4**（L1）。预期占比从 0.6% → ~3-4%。

### 5. fuel_efficiency — **小幅调整**

ep_sum: v2=-16.8 (3.2%) → v3=-27.5 (0.7%)。虽然占比不高，但 -0.05/step 对 892 步 episode 累积到 -27.5。保持轻量即可，**不变**。

---

## 修改决策

- **层级**：L1（纯系数调整）。best=178.70 距目标 200 仅差 ~10.6%，且 v2 框架已证明有效，v3 的 L2 改动反而破坏。
- **语义方向**：回归 v2 per-step landing_contact + 系数再平衡
- **修改组件**：landing_contact（降）、settling_stability（升）、attitude_constraint（升）——三者服务于同一语义：「降低 landing 主导度，让约束信号可听见」

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v4。

    v3→v4 修改（L1 系数再平衡）：
    
    问题诊断（v3）：landing_contact 从 per-step 改为 transition-based (25.0/腿) 后，
    ep_sum 从 285.6 暴涨至 3090（10.8x），magnitude_share 从 54% 升至 80%。
    agent 学会在垫板上弹跳（contact 反复 0→1→0→1）来收割 transition bonus。
    settling_stability 和 attitude_constraint 被完全淹没（合计 <2%）。
    score 从 178.7 暴跌至 88.5。

    v4 方案：回归 v2 的 per-step landing_contact 机制（方向正确），但大幅降低权重：
    - landing_contact: 2.0/leg → 0.5/leg（降低 4x，消除 dominance）
    - settling_stability: -0.3 → -0.4（提升速度约束可听度）
    - attitude_constraint: 角度权重 0.3 → 0.4（提升姿态约束可听度）
    - goal_proximity 和 fuel_efficiency 不变
    
    预期：landing ~15-20%，goal_proximity ~40-45%，settling ~6-8%，attitude ~3-4%。
    """
    
    # ========== 信号提取 ==========
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    # ========== 距离与速度 ==========
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    next_speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    
    # ========== Component A: goal_proximity ==========
    # 改进量（保持 credit assignment 一致性）
    dist_improvement = current_dist - next_dist
    
    # 指数邻近奖励：在目标附近提供每步强梯度
    proximity_bonus = 1.0 * (2.718281828 ** (-next_dist / 0.5))
    
    goal_proximity = 3.0 * dist_improvement + proximity_bonus
    
    # ========== 邻近门控（多个组件共用）==========
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)
    
    # ========== Component B: settling_stability ==========
    # v4: 速度惩罚系数 -0.3 → -0.4，略微增强约束可听度
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.4 * next_speed * settling_multiplier
    
    # ========== Component C: attitude_constraint ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)
    
    # v4: 角度误差权重 0.3 → 0.4，略微增强姿态约束
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.4 * next_angle_err + 0.1 * next_ang_vel_mag)
    
    # ========== Component D: landing_contact (v4: 回归 per-step，降权) ==========
    # v4: 系数从 2.0/leg 降至 0.5/leg。保持 per-step 机制（v2 验证有效），
    # 但大幅降低量级以消除 dominance 和不稳定振荡。
    near_target = next_dist < 0.5
    low_speed = next_speed < 1.0
    
    landing_contact = 0.0
    if near_target and low_speed:
        landing_contact = 0.5 * (next_left_contact + next_right_contact)
    
    # ========== Component E: fuel_efficiency ==========
    fuel_efficiency = -0.05 * float(action != 0)
    
    # ========== 汇总 ==========
    total_reward = (
        goal_proximity +
        settling_stability +
        attitude_constraint +
        landing_contact +
        fuel_efficiency
    )
    
    components = {
        'goal_proximity': goal_proximity,
        'settling_stability': settling_stability,
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency
    }
    
    return float(total_reward), components
```