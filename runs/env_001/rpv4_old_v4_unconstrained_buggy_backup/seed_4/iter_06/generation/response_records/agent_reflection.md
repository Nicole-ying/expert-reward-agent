# Response Record

现在我已经掌握了完整的历史脉络。让我做出诊断并输出修改方案。

---

## 诊断报告

### Evidence

| 证据 | 数据 | 含义 |
|---|---|---|
| **iter_4 → iter_5 退化** | score 155→82, terminated 19/20→1/20 | iter_5 的 transition reward (4.0x) 是灾难性修改 |
| **landing_contact 历史振荡** | iter_3:647(91%) → iter_4:35(41%) → iter_5:479(91%) | 任何试图放大 contact 信号的操作都会导致 exploit |
| **iter_5 checkpoint 趋势** | landing_contact cp70%=411→cp80%=545→cp90%=23→cp100%=509 | 极不稳定，agent 在「狂刷 contact」和「放弃 contact」间剧烈摇摆 |
| **iter_5 score 崩塌** | cp50%=233→cp60%=226→cp70%=113→cp80%=90→cp90%=274→cp100%=86 | 训练中后期严重震荡，无法收敛 |
| **iter_4 是历史最优** | 155.35 / 19/20 terminated | 唯一成功完成任务的版本，结构正确 |
| **但 iter_4 cp80% 仍有 spike** | landing_contact cp80%=151.7→cp90%=26.9 | 连续状态奖励仍可被短期 exploit |

### Behavior Diagnosis

当前策略 (iter_5) 的行为模式是**病态的**：

- agent 学会了在接触区边界来回振荡：进入接触区 → 领取 4.0x transition 奖励 → 离开接触区 → 再次进入 → 再次领取。这解释了 landing_contact 58.4% active_rate 和 479 ep_sum（远超 iter_4 的 35）。
- 只有 1/20 真正完成任务（terminated），其余 19/20 被截断。agent 在"刷奖励"而不是"完成着陆"。
- iter_4 证明：**连续接触奖励（无 base_bonus）+ 适中系数（2.0）已经足以驱动 19/20 完成任务**。问题不在于奖励不够，而在于 iter_5 的 transition 设计创造了更严重的 exploit。

### 干预方案：L2 — potential-based landing reward

**核心思路**：将 landing_contact 从"连续状态奖励"改为"势能差奖励"。

- 定义 `landing_potential = prox_factor × speed_factor × contact_frac`（范围 0~1）
- 只奖励 potential 的**正向增量**：`max(0, potential_next - potential_prev)`
- 单集最大总奖励被严格限制在 `coeff × 1.0`，彻底消除无限积累 exploit
- 使用系数 20.0：单集最大 +20，与 iter_4 的 ep_sum=35 量级接近但更安全
- 其余组件完全沿用 iter_4（已被证明有效）

**为什么不用 iter_5 的 transition 思路**：iter_5 的 `4.0 × contact_gain` 只奖励接触程度的**原始增量**（0→1 给 4.0），但 contact 是二值的（0 或 1），agent 可以通过失去再获得来反复触发。而 potential-based 方法中，一旦 potential 达到 1.0（完美着陆质量），就无法再获得增量——达到天花板后自动免疫 exploit。

---

## 完整代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v6。

    核心改动：landing_contact 从连续状态奖励 → potential-based 势能差奖励。
    - 定义 landing_potential = prox_factor * speed_factor * contact_frac (范围 0~1)
    - 仅奖励 potential 的正向增量：max(0, Δpotential)
    - 单集最大奖励严格 bounded（coeff × 1.0），彻底防止 exploit
    - 一旦达到完美着陆质量（potential=1.0），无法再获得更多 contact 奖励

    设计原则（继承 v4 的成熟结构）：
    - goal_proximity: 二次距离代价 → 引导接近（远处强、近处弱）
    - settling_stability: 二次速度代价 → 引导减速（高速强、低速弱）
    - landing_contact: 势能差奖励 → 奖励"改善着陆质量"而非"保持状态"
    - attitude_constraint: 姿态代价 → 保持平稳
    - fuel_efficiency: 燃料代价 → 节省推力

    Args:
        obs: 当前观测 (8,)
        action: 当前动作 (0-3)
        next_obs: 下一观测 (8,)
        original_reward: 未使用
        info: 未使用
        training_progress: 未使用
    """

    # ========== 信号提取 ==========
    # 上一状态（用于计算 potential 增量）
    prev_x_pos = obs[0]
    prev_y_pos = obs[1]
    prev_x_vel = obs[2]
    prev_y_vel = obs[3]
    prev_left_contact = obs[6]
    prev_right_contact = obs[7]

    # 当前状态（转移后）
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ========== 辅助量 ==========
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5

    prev_distance = (prev_x_pos ** 2 + prev_y_pos ** 2) ** 0.5
    prev_speed = (prev_x_vel ** 2 + prev_y_vel ** 2) ** 0.5

    # proximity_gate: 1.0 在目标处，~0.1 在 distance≈3
    proximity_gate = 1.0 / (1.0 + 3.0 * distance)

    # ========== Component A: goal_proximity — 二次距离代价 ==========
    # 梯度 = -0.2 * distance，远处梯度强，近处梯度弱（天然 curriculum）
    goal_proximity = -0.1 * distance ** 2

    # ========== Component B: settling_stability — 二次速度代价 ==========
    # 梯度随 speed 和 proximity 增强，接近目标时减速约束更强
    settling_stability = -0.3 * speed ** 2 * (0.2 + 0.8 * proximity_gate)
    settling_stability += -0.08 * ang_vel ** 2 * (0.2 + 0.8 * proximity_gate)

    # ========== Component C: attitude_constraint — 姿态代价 ==========
    angle_error = abs(body_angle)
    ang_vel_mag = abs(ang_vel)
    # gate 范围：远距离 0.3，近距离 1.0
    attitude_gate = 0.3 + 0.7 * proximity_gate
    attitude_constraint = -0.4 * angle_error * attitude_gate
    attitude_constraint += -0.15 * ang_vel_mag * attitude_gate

    # ========== Component D: landing_contact — 势能差着陆奖励 ==========
    # v6 核心改动：从连续状态奖励改为 potential-based 势能差奖励。
    #
    # 定义 landing_potential ∈ [0, 1]：
    #   potential = prox_factor × speed_factor × contact_frac
    # 其中 prox_factor = 1 - d/0.5, speed_factor = 1 - s/1.0,
    # contact_frac = (left + right) / 2
    #
    # 仅当 next_obs 满足 near_pad + slow_enough 时计算奖励：
    #   reward = coeff × max(0, potential_next - potential_prev)
    #
    # 这天然防止 exploit：一旦 potential 达到 1.0，无法再获得增量。
    # 单集最大总奖励 = coeff × 1.0，严格 bounded。

    near_pad_next = distance < 0.5
    slow_enough_next = speed < 1.0

    if near_pad_next and slow_enough_next:
        # 当前 potential
        prox_factor_next = 1.0 - distance / 0.5       # 1.0@d=0, 0.0@d=0.5
        speed_factor_next = 1.0 - speed / 1.0          # 1.0@s=0, 0.0@s=1.0
        contact_frac_next = (left_contact + right_contact) / 2.0
        potential_next = prox_factor_next * speed_factor_next * contact_frac_next

        # 上一状态 potential（若不在 zone 内则视为 0）
        near_pad_prev = prev_distance < 0.5
        slow_enough_prev = prev_speed < 1.0
        if near_pad_prev and slow_enough_prev:
            prox_factor_prev = 1.0 - prev_distance / 0.5
            speed_factor_prev = 1.0 - prev_speed / 1.0
            contact_frac_prev = (prev_left_contact + prev_right_contact) / 2.0
            potential_prev = prox_factor_prev * speed_factor_prev * contact_frac_prev
        else:
            potential_prev = 0.0

        # 仅奖励改善
        landing_contact = 20.0 * max(0.0, potential_next - potential_prev)
    else:
        landing_contact = 0.0

    # ========== Component E: fuel_efficiency — 燃料代价 ==========
    fuel_efficiency = -0.03 * float(action != 0)

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
