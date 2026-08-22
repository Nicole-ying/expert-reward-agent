现在我已掌握全局数据。让我逐一诊断：

---

## 诊断报告

### evidence

| 信号 | 数据 |
|---|---|
| **Score** | −121.5，全程平坦（−112 ~ −121），训练10个checkpoint间零改善 |
| **Episode** | len=68.5，19/20 早期终止（<150步 & score<−50），全部 terminated |
| **goal_proximity** | ep_sum=2.149，**10个checkpoint恒定不变**（2.148~2.158），active=100% |
| **settling_stability** | ep_sum=0.44~0.56，缓慢下降，active=100% |
| **attitude_constraint** | ep_sum=−1.4~−1.9，始终为负，active=100%（gate=0.1时仍激活） |
| **landing_contact** | ep_sum=0.3~0.5，active=0.1%~0.2%，几乎从未触发 |
| **fuel_efficiency** | ep_sum=−0.09~−0.24，active=2.6%~6.8% |
| **组件总和** | ≈+1.0/episode，但score=−121.5 → 环境原生惩罚约−122/episode 主导 |

### behavior_diagnosis

当前奖励函数存在**三个根本性缺陷**，导致 agent 完全无法学习：

**1. 纯 improvement-delta 设计导致梯度消失（致命）**

`goal_proximity` 使用 `2.0 * (current_dist - next_dist)`，这是一个**可 telescoping 的形式**：累计和 = `2.0 × (初始距离 − 最终距离)`。由于 agent 每次都以相似的轨迹 crash，初始距离和 crash 距离几乎恒定，因此 ep_sum 在所有 checkpoint 中完全平坦（2.149 ± 0.01）。**RL 无法优化一个恒定的信号**——它不提供任何关于「哪个状态更好」的梯度信息。同理，`settling_stability` 也是 delta 形式，同样缺乏逐步梯度。

**2. 所有组件量级过小，被环境原生惩罚淹没**

五个组件 ep_sum 合计约 +1.0，但 score 为 −121.5。这意味着环境每步约 −1.8 的原生惩罚主导了整个训练信号。agent 收到的有效奖励 ≈ 环境惩罚，我们的奖励函数几乎不起作用。

**3. 关键组件未激活**

`landing_contact` 的激活率仅 0.1%~0.2%——条件 `near_target AND low_speed AND both_contacts` 过于严格。`attitude_constraint` 的 gate 默认为 0.1（而非 0），导致姿态惩罚始终激活，agent 在所有阶段都被惩罚角度偏差，无法区分「远距离大角度机动」（合理）和「近目标倾斜」（危险）。

**结论**：当前奖励函数需要从根本上重写——从纯 delta 形式切换为 **delta + 直接状态惩罚混合架构**，大幅提升量级，修复门控逻辑。

---

## 新版奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v2。
    
    根本性改动（vs v1）：
    - 从纯 improvement-delta 切换为「delta + 直接状态惩罚」混合架构，
      确保每一步都有可优化的梯度信号。
    - 量级整体提升 5~10 倍，使我们的奖励不被环境原生惩罚淹没。
    - 姿态约束门控修复：远距离时完全关闭（gate=0），不再无故惩罚大角度机动。
    - landing_contact 条件放宽 + 奖励提升，使 agent 能感知到着陆信号。
    """
    
    # ========== 信号提取 ==========
    # 当前状态
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]
    
    # 距离与速率（L2 范数）
    current_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    current_speed = (vx**2 + vy**2) ** 0.5
    next_speed = (nvx**2 + nvy**2) ** 0.5
    
    # ================================================================
    # Component A: goal_proximity — 混合架构（delta + 直接状态惩罚）
    # ================================================================
    # delta 部分：奖励每一步向目标靠近的进展
    dist_improvement = current_dist - next_dist
    
    # 直接状态惩罚：距离越远惩罚越重，提供逐步梯度
    # 使用平滑函数：dist=0→0，dist=1→−1.5，dist=2→−3.0
    dist_penalty = -1.5 * next_dist
    
    goal_proximity = 8.0 * dist_improvement + dist_penalty
    
    # ================================================================
    # Component B: settling_stability — 速度管理（delta + 直接惩罚）
    # ================================================================
    # 接近度门控：距离越近，速度控制越重要
    # dist=0 → gate≈1.0，dist=1 → gate≈0.33，dist=2 → gate≈0.2
    prox_gate = 1.0 / (1.0 + 2.0 * next_dist)
    
    # delta：奖励减速
    speed_improvement = current_speed - next_speed
    
    # 直接速度惩罚（仅在接近目标时显著）
    speed_penalty = -4.0 * next_speed * prox_gate
    
    settling_stability = 2.0 * speed_improvement * (0.5 + prox_gate) + speed_penalty
    
    # ================================================================
    # Component C: attitude_constraint — 姿态管理（修复门控）
    # ================================================================
    # 两级门控：
    #   dist >= 1.5  → gate=0（完全不干预，允许大角度机动）
    #   dist <= 0.5  → gate=1（全力约束姿态）
    #   中间线性过渡
    if next_dist >= 1.5:
        att_gate = 0.0
    elif next_dist <= 0.5:
        att_gate = 1.0
    else:
        att_gate = 1.0 - (next_dist - 0.5) / 1.0  # 从 1.0 线性降到 0.0
    
    # delta 部分
    angle_improvement = abs(angle) - abs(n_angle)
    angvel_improvement = abs(ang_vel) - abs(n_ang_vel)
    
    # 直接姿态惩罚
    angle_penalty = -att_gate * (abs(n_angle) * 3.0 + abs(n_ang_vel) * 1.5)
    
    # delta 奖励
    angle_delta_reward = att_gate * (2.0 * angle_improvement + 1.0 * angvel_improvement)
    
    attitude_constraint = angle_delta_reward + angle_penalty
    
    # ================================================================
    # Component D: landing_contact — 安全着陆奖励
    # ================================================================
    # 放宽条件，使用连续信号而非硬门控
    near_target = next_dist < 0.3
    low_speed = next_speed < 0.5
    both_contacts = (n_left > 0.5 and n_right > 0.5)
    any_contact = (n_left > 0.5 or n_right > 0.5)
    
    landing_contact = 0.0
    if near_target and low_speed:
        if both_contacts:
            landing_contact = 10.0  # 完美双支撑着陆，每步大奖励
        elif any_contact:
            landing_contact = 2.0   # 单边接触，小鼓励
        # 接近且低速但无接触：不给奖励，不惩罚（让 settling 自然引导）
    
    # ================================================================
    # Component E: fuel_efficiency — 燃料效率
    # ================================================================
    fuel_efficiency = -0.05 * float(action != 0)
    
    # ================================================================
    # 汇总
    # ================================================================
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
        'fuel_efficiency': fuel_efficiency,
    }
    
    return float(total_reward), components
```

---

### v1 → v2 变更摘要

| 维度 | v1（iter_1） | v2（本轮） | 理由 |
|---|---|---|---|
| **架构** | 纯 improvement-delta | delta + 直接状态惩罚混合 | delta 可 telescoping，梯度消失；直接惩罚每步都有梯度 |
| **goal_proximity 量级** | 2.0× delta | 8.0× delta + (−1.5× dist) | 原来 ep_sum≈2 恒定不变；现在每步都有 −1.5×dist 的梯度推动 |
| **settling 直接惩罚** | 无 | −4.0×speed×prox_gate | 原来只有 delta，无法持续压制高速；现在接近目标时强压速度 |
| **attitude gate** | `0.1 if far else 1.0`（始终激活） | `0.0 if dist≥1.5`，线性过渡 | 原来远距离也被惩罚角度，干扰大角度机动探索 |
| **attitude 直接惩罚** | 无 | −gate×(3|angle|+1.5|angvel|) | 不仅要改进，还要维持在低角度状态 |
| **landing_contact 奖励** | 3.0（极难触发） | 10.0（双接触）/ 2.0（单接触） | 原来 active=0.1%，加强信号并增加中间态引导 |
| **量级整体** | ep_sum≈1 | 预期 ep_sum≈20~50 | 不被环境原生 −122/ep 淹没 |