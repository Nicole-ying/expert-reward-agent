现在我对全局有了完整把握。让我整理诊断和干预方案。

---

## 1. 诊断报告

**evidence**：
- iter1→iter2：得分 -32→107（+139），terminated 0/20→11/20。上一轮的 velocity_penalty 门控修改**非常成功**，agent 学会了着陆。
- 但新问题出现：**contact_bonus magnitude_share = 92.5%**（iter1 时该组件完全死亡，magnitude_share=0%）。
- checkpoint 轨迹揭示了因果关系：contact_bonus 在 cp40%、cp60%、cp100% 分别激活（ep_sum=1.8, 156, 183），**得分完全跟随 contact_bonus 波动**——它激活时得分骤升，不激活时得分为负或接近零。agent 不是稳定学会了着陆，而是在随机探索中偶尔碰到正确着陆姿态获得巨额奖励。
- goal_proximity 从 iter1 的 magnitude_share=47.8% 被挤压到 iter2 的 2.5%，连续引导信号完全被淹没。
- velocity_penalty 从 -0.49→-5.31（+10x），说明指数高度门控生效了，但它 magnitude_share 也只有 3.2%，同样被淹没。
- orientation_penalty magnitude_share 从 46.5%→1.9%。

**behavior_diagnosis**：agent 的行为由稀疏的 contact_bonus（仅在着陆瞬间激活 ~22 步，每步 ~7.0）主导。在 775 步中，连续引导信号（goal_proximity + velocity_penalty + orientation_penalty）合计仅约 -4.4/episode，而 contact_bonus 贡献 +154/episode。agent 缺乏从远处到近处的平滑梯度，着陆成功与否高度随机（cp 得分剧烈波动证明了这一点）。

**signal_completeness**：所有必要职责已覆盖（接近引导、速度阻尼、姿态惩罚、着陆奖励），但**信号强度严重失衡**。contact_bonus 吸收了 92.5% 的信号能量，使连续引导信号形同虚设。

**selected_level**：**L1**。核心架构正确（四组件骨架在 iter2 验证有效，得分大幅提升），只需调权重恢复信号平衡。上一轮大改 velocity_penalty 是必要的（从死组件到激活），本轮只需微调系数。且差距尚大（107→200），不适合冒险 L2/L3。

**selected_intervention**：
- **contact_bonus**：`w_contact` 从 10.0 → 4.0（降低 60%）
- **goal_proximity**：`w_x` 从 1.0 → 2.0，`w_y` 从 2.0 → 4.0（翻倍）
- 两个调整服务于同一语义：**恢复稀疏着陆奖励与连续引导信号之间的量级平衡**。

**falsifiable_hypothesis**：降低 contact_bonus 使其不再主导 reward 后，goal_proximity 的连续梯度将能有效引导 agent 从远处向目标靠近；同时 contact_bonus 保持足够强度（~4.0/步 × ~22 步 ≈ 88/episode）仍能提供着陆正反馈。预期 agent 轨迹更平滑、着陆更一致（terminated 比例上升），最终得分向 200 靠近。

**expected_next_round**：
- contact_bonus magnitude_share 从 92.5% 降至 55-75%
- goal_proximity magnitude_share 从 2.5% 升至 10-20%
- terminated 比例从 11/20 上升（连续引导更强 → 更稳定到达目标区域）
- checkpoint 得分轨迹波动减小（不再完全依赖 contact_bonus 的随机激活）
- 整体得分应上升，因为更好的连续引导 → 更高效到达着陆区 → 更多成功着陆

**main_risk**：如果 contact_bonus 降得太多而 goal_proximity 仍不足以提供强梯度，agent 可能失去着陆动机，terminated 比例反而下降。但 iter1→iter2 的经验表明 velocity_penalty（高度门控减速）已经在引导近地减速行为，goal_proximity 翻倍后应能承担更多引导职责。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3 reward: rebalanced — reduced contact_bonus dominance, stronger goal_proximity

    Key change from v2: contact_bonus weight reduced 10→4 (was 92.5% magnitude share),
    goal_proximity weights doubled (w_x 1→2, w_y 2→4) to restore continuous guidance.
    This balances the sparse landing bonus against the continuous approach signals.
    
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

    # --- 1. goal_proximity (improvement_delta, weights doubled) ---
    # encourage moving toward (0,0) by reducing squared distance
    # larger weight on vertical axis to avoid hard landings
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2   # positive when approaching horizontally
    delta_y_sq = y_cur ** 2 - y_nxt ** 2   # positive when approaching vertically
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    # Strong penalty for high speed only when close to the ground.
    # Gate = exp(-y_nxt / h_scale): ~1.0 at y=0, ~0.37 at y=1, ~0.05 at y=3.
    # This lets the agent fly fast at high altitude but forces deceleration for landing.
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. safe_contact_bonus (joint condition proxy, weight reduced) ---
    # reward when both legs touch and position is near (0,0)
    # weight reduced from 10.0 to 4.0 to prevent dominance over continuous signals
    w_contact = 4.0
    y_th = 0.1   # within 0.1 vertical distance
    x_th = 0.2   # within 0.2 horizontal offset
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt   # 0.0 or 1.0
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)
```