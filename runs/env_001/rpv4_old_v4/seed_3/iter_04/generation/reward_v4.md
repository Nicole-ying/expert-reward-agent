Now I have all the data. Let me compile the full diagnosis.

---

## 1. 诊断报告

### evidence

**三轮回溯**：

| iter | score | term | contact_bonus | 主导问题 |
|------|-------|------|--------------|---------|
| 1 | -32.35 | 0/20 | DEAD (0%) | velocity_penalty 太弱 (-0.486)，稀疏 contact 从未被发现 |
| 2 | **107.13** | **11/20** | ep_sum=154.47, active=2.8%, magnitude=**92.5%** | 改了 velocity_penalty 形态（曼哈顿→指数高度门控），agent 学会了着陆，但 contact_bonus 完全主导 |
| 3 | -26.17 | 0/20 | DEAD (0%) | 降低 contact_bonus(10→4)、加倍 goal_proximity 权重(1→2, 2→4)，着陆行为完全丧失 |

**Checkpoint 轨迹（iter 3）揭示了关键证据**：
- cp60% 时 score=**207.1**，contact_bonus 短暂激活 (5.8%, ep_sum=105.7)
- cp70% 时 score=84.2，contact_bonus 仍有 4.2% 激活
- cp80% 时 contact_bonus **归零**，score 跌至 -12.1
- 这说明 agent **能够发现着陆**（cp60-70%），但随后**丧失了该行为**——典型的训练不稳定：稀疏 contact 信号一旦丢失就无法恢复，而连续信号（goal_proximity 加倍后）提供的「hover near (0,0)」局部最优足以维持生存。

**iter 2 vs iter 3 对比**：唯一的区别是 contact_bonus 权重（10→4）和 goal_proximity 权重（w_x 1→2, w_y 2→4）。iter 2 用 w_contact=10 强行突破了稀疏信号的发现瓶颈（尽管导致 92.5% 主导）；iter 3 削弱了该信号后，agent 退化为 hover 策略。

### behavior_diagnosis

Agent 当前策略：**在目标区域附近悬停/漂移，但从不执行最终着陆**。证据：
- len=1000（全部 truncate），0 termination
- goal_proximity ep_sum=7.685（agent 确实在接近 (0,0)）
- 但 contact_bonus=0（从未触地）
- velocity_penalty + orientation_penalty ≈ -8.8，goal_proximity ≈ +7.7，净值 ≈ -1.1，解释了 score ≈ -26

根因：**稀疏 contact_bonus 是唯一奖励着陆的信号，但 w=4 不足以被发现**。当 agent 到达 y≈0.1-0.3 时，goal_proximity 的 delta_y_sq 梯度已经极其微弱（y² 在低空变化很小），没有任何信号推动 agent 完成最后 0.1-0.3 的下降。agent 学到的「最优」策略是：hover 在低空，不触地。

### signal_completeness

| 职责 | 状态 | 问题 |
|------|------|------|
| goal_proximity（接近目标） | ✅ 活跃 | 低空梯度不足（平方距离在 y→0 时导数→0） |
| velocity_damping（近地减速） | ✅ 活跃 | 正常工作 |
| orientation_penalty（姿态稳定） | ✅ 活跃 | 正常工作 |
| **着陆引导（从「接近」到「触地」）** | ❌ **缺失** | contact_bonus 是唯一着陆信号，但太稀疏（3 轮中 2 轮死亡）；缺少连接「hover 在低空」和「触地」之间的连续梯度 |

核心缺口：**没有一个组件提供「从低空到着陆」的连续梯度**。goal_proximity 的平方距离在 y<0.3 时梯度接近零，contact_bonus 是稀疏事件。

### selected_level: **L2**

理由：不是调权重（L1 已经在 iter 2↔3 之间试过 contact_bonus 10→4，无效），也不是换框架（骨架本身在 iter 2 证明可行）。需要一个新的数学形态——**将着陆引导从稀疏事件变为连续过程奖励**。

### selected_intervention

**新增组件 `descent_incentive`**：当 agent 在着陆区附近（y<0.8 且 |x|<0.5）时，奖励向下的位移（y_cur - y_nxt）。

这是一个**全新语义**——不是「检测接触事件」，而是「奖励下降动作」。与 contact_bonus 的语义完全不同：
- contact_bonus：「你到达了目的地」→ 稀疏，难以发现
- descent_incentive：「你正在朝目的地下降」→ 连续，每步都有梯度

保留 contact_bonus（w=4.0）不变，因为一旦 agent 在 descent_incentive 引导下到达 y≈0，contact_bonus 提供终端确认奖励。

### falsifiable_hypothesis

如果 descent_incentive 提供了从低空到着陆的连续梯度，agent 应该能够：
1. 像 iter 2 一样发现着陆行为
2. 但不会像 iter 2 那样被 contact_bonus 主导（因为 descent_incentive 分散了着陆相关奖励到多个时间步）
3. 训练稳定性提高（连续梯度比稀疏 spike 更难被遗忘）

### expected_next_round

- **terminated > 0**：至少部分 episode 成功着陆
- **descent_incentive active_rate > 30%**：在着陆区附近活跃
- **descent_incentive ep_sum > 1.0**：提供有意义的梯度
- **contact_bonus active_rate > 2%**：着陆事件被触发
- **contact_bonus magnitude_share < 60%**：不再主导（因为有 descent_incentive 分担）
- **score 显著改善**：目标 > 0，理想情况 > 50

### main_risk

agent 可能学会「在低空振荡下降-上升」来反复获取 descent_incentive 奖励。缓解因素：
- 上升时不触发奖励（descending=0）
- velocity_penalty 和 orientation_penalty 惩罚不必要的机动
- contact_bonus 提供一次性大奖励，使「着陆并终止」比「振荡」更优
- 如果振荡发生，下一轮可通过收紧 near_ground 门控或添加 oscillation 惩罚来修复

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4 reward: added descent_incentive — continuous gradient for final descent phase

    Key change from v3: new `descent_incentive` component rewards downward progress
    when the agent is near the landing zone (y < 0.8, |x| < 0.5). This bridges the
    gap between "hovering near target" (where goal_proximity gradient vanishes at
    low y due to squared-distance form) and "committing to touchdown."
    
    Contact_bonus is kept at w=4.0 as a terminal landing confirmation, now reachable
    thanks to the continuous approach gradient provided by descent_incentive.
    
    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement_delta, squared distance) ---
    # Encourage moving toward (0,0). Larger weight on vertical axis to avoid hard landings.
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive when approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive when approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    # Strong penalty for high speed only when close to the ground.
    # Gate = exp(-y_nxt / h_scale): ~1.0 at y=0, ~0.37 at y=1, ~0.05 at y=3.
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. descent_incentive (NEW — continuous descent gradient near landing zone) ---
    # Reward downward progress when near the landing pad.
    # Gates ensure this only fires when the agent is in position to land:
    #   near_ground: linearly decays from 1 at y=0 to 0 at y>=0.8
    #   near_center: linearly decays from 1 at x=0 to 0 at |x|>=0.5
    #   descending:  positive only when moving downward (y_cur > y_nxt)
    # This provides a continuous gradient that peaks during the final descent,
    # naturally tapering to zero once landed (y≈0, no further descent possible).
    w_desc = 2.0
    near_ground = max(0.0, 1.0 - y_nxt / 0.8)       # active below y=0.8
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)  # active within |x|<0.5
    descending = max(0.0, y_cur - y_nxt)             # positive when going down
    descent_incentive = w_desc * near_ground * near_center * descending

    # --- 5. contact_bonus (sparse landing event bonus) ---
    # Reward when both legs touch and position is near (0,0).
    # Kept at moderate weight — descent_incentive now provides the approach gradient.
    w_contact = 4.0
    y_th = 0.1   # within 0.1 vertical distance
    x_th = 0.2   # within 0.2 horizontal offset
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt   # 0.0 or 1.0
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + descent_incentive + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'descent_incentive': descent_incentive,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)
```