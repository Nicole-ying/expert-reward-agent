# Response Record

## 诊断报告

### 1. 上一轮改善归因（iter_1 → iter_2: +349分）

**什么修改促成了改善？** 从 improvement_delta（势能差框架）切换到 dense state-based attraction reward（密集状态型吸引奖励）。

**证据：**
- `goal_proximity` ep_sum 从 2.15 (iter_1) → 1238.45 (iter_2)，提升了 576 倍。v1 的 `distance_improvement` 只在距离缩小时才给奖励，梯度稀疏；v2 的 `2.0/(0.3+next_distance)` 每步都给奖励，形成连续梯度场。
- `episode_length` 从 68.5 → 409.10。v1 中 agent 迅速 crash（无有效引导），v2 中存活并完成任务。
- Checkpoint 趋势显示 cp10% 时 goal_proximity 已达 182.98，说明 dense 信号从一开始就提供了有效梯度。

### 2. 当前行为诊断

| 维度 | 观察 | 判断 |
|---|---|---|
| **任务完成** | score=227.77 > 目标200，terminated=20/20 | ✅ 主任务已解决 |
| **episode 长度** | len=409.10，全部自然终止 | 略长，存在优化空间 |
| **目标接近** | goal_proximity 占 86.8% magnitude_share | 极度主导，但这是主驱动的合理代价 |
| **着陆接触** | landing_contact active_rate=11.9%，仅在 cp80%+ 出现 | 着陆信号正常，作为突破 plateaus 的催化剂 |
| **稳定停靠** | settling_stability=-23.35，全程负贡献 | 门控惩罚始终非零，持续微量消耗 |
| **燃料效率** | fuel_efficiency=-8.68，active_rate=63.2% | 63% 的 step 使用引擎，次要目标有提升空间 |
| **cp60% 平台期** | goal_proximity 飙至 2616，score 仍为 -16.4 | 出现过"中距离徘徊"（proxy plateau），但 cp80% 突破 |

**行为模式**：agent 学会了"先接近目标区域 → 收集 proximity 奖励 → 最终触发双足着陆"。这是健康的渐进学习路径。cp60% 的平台期是正常的探索-利用过渡，最终由 landing_contact 奖励突破。

### 3. 信号完备性

| 职责 | 状态 | 评价 |
|---|---|---|
| goal_proximity | ✅ 密集状态型 | 有效，但量级过大（86.8%） |
| settling_stability | ✅ 门控速度惩罚 | 持续工作，但门控在远距离仍非零 |
| attitude_constraint | ✅ 门控姿态惩罚 | 正常工作，量级小 |
| landing_contact | ✅ 门控+转移事件 | 成功催化突破，量级合理 |
| fuel_efficiency | ⚠️ 偏弱 | -0.03/step，对 63% 引擎使用率的抑制力不足 |

### 4. 修改方案

- **selected_level**: **L1** — 当前 score 已超过目标（227.77 > 200），差距 > 10% 但方向正确。根据规则"接近目标时优先 L1"，仅调整系数不改数学形态。
- **selected_intervention**: **fuel_efficiency 惩罚系数** 从 `-0.03` 提升到 `-0.05`（与 v1 对齐）。
- **falsifiable_hypothesis**: 当前 agent 已掌握主任务，适度的燃料惩罚会促使其减少不必要的引擎使用（如远距离时的微调喷射），同时不影响已习得的"接近→着陆"核心行为。因为 goal_proximity 量级远大于 fuel_efficiency（1238 vs -8），即使惩罚翻倍，主信号仍占绝对主导。
- **expected_next_round**: 
  - fuel_efficiency ep_sum 预计升至约 -14~-16（与引擎使用率相关）
  - active_rate 预计从 63.2% 小幅下降
  - score 预计仍保持在 200 以上
  - episode_length 可能略有缩短（减少无意义喷射 → 更早着陆）

### 5. 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v3。
    
    框架保持 v2 的密集状态型吸引奖励。
    L1 修改：fuel_efficiency 惩罚从 -0.03 提升到 -0.05，
    加强次要目标「最小化引擎推力使用」，对齐 v1 中的惩罚力度。
    
    Args:
        obs: 当前观测 (8,) [x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, left_contact, right_contact]
        action: 当前动作 (0-3: no_engine, left_orient, main_engine, right_orient)
        next_obs: 下一观测 (8,)
        original_reward: 原始奖励（不使用）
        info: 信息字典（为空，不使用）
        training_progress: 训练进度（v3不使用）
    
    Returns:
        float: 总奖励
        dict: 各奖励组件
    """

    # ========== 信号提取 ==========
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

    # ========== 辅助量 ==========
    # 到目标垫板的欧氏距离
    current_distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5

    # 下一状态的合速度（标量）
    next_speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5

    # 接近度门控：距离越近值越高，用于激活近目标约束
    # exp(-2*d): d=0→1, d=0.5→0.37, d=1→0.14, d=2→0.018, d=3→0.002
    proximity_gate = 2.718281828 ** (-2.0 * next_distance)

    # ========== Component A: 目标接近（密集状态型吸引奖励） ==========
    # role: goal_proximity
    # 每步独立计算：距离越近奖励越高，提供连续梯度。
    # scale/(offset+distance): d=0→6.67, d=0.5→2.5, d=1→1.54, d=2→0.87, d=3→0.61
    state_based_reward = 2.0 / (0.3 + next_distance)

    # 小幅 improvement bonus：直接奖励「距离减小」的步，强化即时因果
    distance_improvement = current_distance - next_distance
    improvement_bonus = 1.5 * max(0.0, distance_improvement)

    goal_proximity = state_based_reward + improvement_bonus

    # ========== Component B: 稳定停靠（门控速度惩罚） ==========
    # role: settling_stability
    # 仅在接近目标时激活，鼓励减速停靠。
    # 远距离时 proximity_gate≈0，完全不干预自由飞行。
    settling_stability = -proximity_gate * next_speed * 0.8

    # ========== Component C: 姿态约束（门控绝对惩罚） ==========
    # role: attitude_constraint
    # 仅在接近目标时要求姿态平稳（角度小 + 角速度低）。
    attitude_constraint = -proximity_gate * (
        abs(next_body_angle) * 0.5 + abs(next_ang_vel) * 0.3
    )

    # ========== Component D: 安全着陆接触（门控 + 转移事件） ==========
    # role: landing_contact
    near_target = next_distance < 0.3
    both_contacts = (next_left_contact > 0.5 and next_right_contact > 0.5)
    prev_both_contacts = (left_contact > 0.5 and right_contact > 0.5)

    landing_contact = 0.0
    if near_target and both_contacts:
        # 维持双接触的持续奖励
        landing_contact = 3.0
        # 首次获得双接触的额外奖励（状态转移事件）
        if not prev_both_contacts:
            landing_contact += 5.0

    # ========== Component E: 燃料效率（L1: 加强惩罚） ==========
    # role: fuel_efficiency
    # 改动：惩罚系数从 -0.03 提升到 -0.05，对齐 v1 力度。
    # 原因：agent 已掌握主任务（score=227.77>200），
    # 适度加强燃料约束以推进次要目标，不破坏主行为。
    fuel_efficiency = -0.05 * float(action != 0)

    # ========== 汇总奖励 ==========
    total_reward = (
        goal_proximity
        + settling_stability
        + attitude_constraint
        + landing_contact
        + fuel_efficiency
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
