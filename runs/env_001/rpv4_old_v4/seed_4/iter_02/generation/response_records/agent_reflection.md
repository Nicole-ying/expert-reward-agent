# Response Record

---

## 1. 诊断报告

### evidence

| 证据来源 | 关键数据 |
|---|---|
| `read_memory` | iter_1 score=-121.51, len=68.5, terminated=20/20，唯一迭代 |
| `read_training_feedback` | 组件 ep_sum 总和 ≈ +1.05，但 score=-121.5 → 环境终止惩罚 ≈ **-122** 碾压一切 |
| `read_checkpoint_trend` — score | cp10%=-113.9 → cp100%=-121.3，**全程平坦，零学习进展** |
| `read_checkpoint_trend` — goal_proximity | cp10%=2.149 → cp100%=2.149，**完全冻结**。improvement_delta 框架下总奖励被初始距离锁定，无论 agent 做什么，总和不变 |
| `read_checkpoint_trend` — attitude_constraint | 始终为负（-1.4~-1.9），active_rate=100%。improvement-based 公式在飞行器自然角振荡中持续产生负值 |
| `read_checkpoint_trend` — landing_contact | active_rate=0.1%-0.2%，条件过严（\|x\|<0.2, \|y\|<0.2, speed<0.5），agent 在到达前已 crash |
| early_terminal | 19/20 集在 150 步内以 score<-50 终止 |

### behavior_diagnosis

Agent 没有学到任何有效行为。训练曲线完全平坦（从 10% 到 100% 无改善），说明 **RL 优化器从奖励函数中提取不到有意义的梯度**。所有 episode 在 ~68 步 crash 终止。当前的 improvement_delta 框架存在根本性缺陷：

- `goal_proximity = 2.0 * (current_distance - next_distance)`：这是势能差奖励，**总奖励被严格限制在 `2.0 × 初始距离`**，无论 agent 表现如何。初始距离≈1-2，总奖励上限≈2-4，远小于 -122 的终止惩罚。
- `attitude_constraint` 用 improvement-based 公式 + gate=0.1（远距离时），飞行器的自然角振荡使 `angle_improvement` 期望为负，产生持续 drain。
- `landing_contact` 的条件门过严，agent 永远触发不了，形同虚设。

### signal_completeness

| 必要职责 | 状态 | 问题 |
|---|---|---|
| goal_proximity（导航引导） | **无效** | improvement_delta 总量有界，信号淹没在终止惩罚中 |
| settling_stability（减速） | 弱 | improvement-based，远距离时权重太低 |
| attitude_constraint（姿态） | **有害** | 全程持续负 drain，不区分阶段 |
| landing_contact（着陆） | **死亡** | active_rate=0.2%，条件太严苛 |
| fuel_efficiency（节油） | OK | 轻量惩罚，不是主问题 |

### selected_level: L3

**选择理由**：score=-121 且完全平坦（10 个 checkpoint 无任何改善趋势），符合「score 始终接近零或负，当前奖励函数从根上错了」的 L3 触发条件。improvement_delta 框架的数学上限（势能差有界）与任务需求（需克服 ~-122 的终止惩罚）存在根本性矛盾，必须换主信号框架。

### selected_intervention

**新框架：密集状态型吸引奖励 + 门控约束**

核心改动（全部服务于同一语义——「让奖励函数产生足够强的梯度引导 agent 到达目标」）：

1. **goal_proximity（L3 重写）**：从 `improvement_delta` 切换为 `scale / (offset + distance)` 状态型密集奖励 + 小幅 improvement bonus。状态型奖励每步独立计算，总量不再被初始距离锁死，距离越近奖励越高，提供清晰连续的梯度。
   
2. **attitude_constraint（L2 重写）**：从 improvement-based 切换为门控绝对惩罚 `-prox_gate * (|angle| + |ang_vel|)`。远距离时 `prox_gate ≈ 0`，完全关闭；只有接近目标时才激活。消除持续 drain。

3. **settling_stability（L2 重写）**：从 improvement-based 切换为门控速度惩罚 `-prox_gate * speed`。同样只近目标时激活。

4. **landing_contact（L2 调整）**：放宽 near_target 阈值 0.2→0.3，增加首次接触 bonus，提高总奖励上限。

5. **fuel_efficiency**：保持轻量惩罚不变。

### falsifiable_hypothesis

状态型密集奖励 `2.0/(0.3+distance)` 在距离=2 时约 0.87/步、距离=0.3 时约 3.33/步，提供每步清晰的正梯度。agent 随机探索时也会因偶然靠近目标而获得更高奖励，从而建立「靠近目标=好」的关联。门控约束只在近目标时激活，不干扰早期探索。预计 agent 将开始向目标移动，score 从 -121 显著回升。

### expected_next_round

- score 预计从 -121 改善到 -80~-30 区间（仍可能为负，因为终止惩罚仍存在）
- goal_proximity 的 ep_sum_mean 和 active_rate 将随 agent 接近目标而显著上升
- settling_stability 和 attitude_constraint 的 magnitude 在远处接近零，在近处适度激活
- landing_contact 的 active_rate 从 0.2% 提升到可观测水平
- 如果 agent 能到达 landing zone，可能首次出现成功着陆的 episode

---

## 2. 奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v2。
    
    框架变更（L3）：从 improvement_delta（势能差）切换为密集状态型吸引奖励。
    核心思路：每步独立给予距离相关奖励，提供清晰连续梯度，突破势能差框架的总量上限。
    
    Args:
        obs: 当前观测 (8,) [x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, left_contact, right_contact]
        action: 当前动作 (0-3: no_engine, left_orient, main_engine, right_orient)
        next_obs: 下一观测 (8,)
        original_reward: 原始奖励（不使用）
        info: 信息字典（为空，不使用）
        training_progress: 训练进度（v2不使用）
    
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

    # ========== Component A: 目标接近（L3：密集状态型吸引奖励） ==========
    # role: goal_proximity
    # 核心改动：不再使用 improvement_delta，改用状态型公式。
    # 每步独立计算：距离越近奖励越高，提供连续梯度。
    # scale/(offset+distance): d=0→6.67, d=0.5→2.5, d=1→1.54, d=2→0.87, d=3→0.61
    state_based_reward = 2.0 / (0.3 + next_distance)

    # 小幅 improvement bonus：直接奖励「距离减小」的步，强化即时因果
    distance_improvement = current_distance - next_distance
    improvement_bonus = 1.5 * max(0.0, distance_improvement)

    goal_proximity = state_based_reward + improvement_bonus

    # ========== Component B: 稳定停靠（L2：门控速度惩罚） ==========
    # role: settling_stability
    # 改动：从 improvement-based 切换为门控绝对速度惩罚。
    # 仅在接近目标时激活，鼓励减速停靠。
    # 远距离时 proximity_gate≈0，完全不干预自由飞行。
    settling_stability = -proximity_gate * next_speed * 0.8

    # ========== Component C: 姿态约束（L2：门控绝对惩罚） ==========
    # role: attitude_constraint
    # 改动：从 improvement-based 切换为门控绝对惩罚。
    # 消除 v1 中全程持续的负 drain。
    # 仅在接近目标时要求姿态平稳（角度小 + 角速度低）。
    attitude_constraint = -proximity_gate * (
        abs(next_body_angle) * 0.5 + abs(next_ang_vel) * 0.3
    )

    # ========== Component D: 安全着陆接触（L2：放宽条件 + 增强奖励） ==========
    # role: landing_contact
    # 改动：near_target 阈值从 0.2 放宽到 0.3，增加首次双接触 bonus。
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

    # ========== Component E: 燃料效率（保持轻量） ==========
    # role: fuel_efficiency
    # 对任何引擎使用施加轻微惩罚
    fuel_efficiency = -0.03 * float(action != 0)

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
