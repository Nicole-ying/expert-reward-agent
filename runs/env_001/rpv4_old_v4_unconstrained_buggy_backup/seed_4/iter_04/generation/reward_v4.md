Now I have the complete picture. Let me compile the evidence and diagnosis.

---

## 诊断报告

### Evidence

| 证据 | 数据 | 含义 |
|---|---|---|
| `landing_contact` 575x 暴涨 | iter_2: 1.1 → iter_3: 647.2，magnitude_share=91.2% | iter_3 的软化门控 + base_bonus 创造了巨大的 exploit 漏洞 |
| `landing_contact` active_rate=88% | 880/1000 步在领取奖励 | agent 学会了长时间滞留在 reward zone 而非完成任务 |
| Score 训练中期 peak→衰退 | cp50%: 104.7 → cp80%: 3.3 → cp100%: -27.0 | 典型的 over-optimize 模式：agent 在 cp50% 后学会 exploit landing_contact，总分数反而下降 |
| landing_contact checkpoint 趋势 | cp30%: 292 → cp80%: 724 → cp100%: 588 | cp30% 后暴涨，与 score 从 cp50% 开始衰退的时间线吻合 |
| 20/20 truncated, 0/20 terminated | len=1000，无 crash 也无成功 | agent 学会了安全地徘徊，但从未真正完成着陆 |
| `goal_proximity` 迭代趋势 | iter_1: +2.1 → iter_2: -33.3 → iter_3: -16.4 | 从 improvement→linear→quadratic 后量级缩小到合理范围，但被 landing_contact 完全淹没（share 仅 2.3%） |
| 所有负向组件在 iter_3 的 share | goal_proximity 2.3%, settling 0.7%, attitude 1.7%, fuel 4.1% | 四个负向组件合计仅 8.8%，对 agent 决策几乎无影响 |

### Behavior Diagnosis

**当前策略：精准的"hover exploit"**。agent 学会了停留在目标垫板附近（distance≈0.42）、保持低速（speed≈0.17）、但**不完成着陆**。它在 `near_pad & slow_enough` 区域内反复领取 `base_bonus`（最高 1.5/步），积累了 647 的 `landing_contact` 奖励。这个奖励如此巨大（91.2% share），导致所有其他信号——包括真正的任务完成所需的减速、姿态控制、燃料节省——都被淹没。agent 发现"在垫板附近飘着"比"完成着陆"更划算：一旦着陆成功 episode 就终止了，不如一直飘着持续领奖。

**这是不健康的**：cp50% 时 score=104.7 证明 agent 有能力学得更好，但 landing_contact 的 exploit 把策略拖向了局部最优。

### 根因

`base_bonus = 1.5 * prox_factor * speed_factor` 是罪魁祸首。它奖励"靠近+低速"而不要求实际接触，属于**持续事件被重复领取**的经典模式。agent 不需要完成着陆就能无限收割这个奖励。

### 干预方案

**L2 — 持续→接触事件**：移除 `base_bonus`，`landing_contact` 仅在实际有腿接触时发放。这样 agent 无法在不接触的情况下获得奖励，"hover exploit" 被阻断。`goal_proximity` 负责引导接近，`settling_stability` 负责引导减速，`landing_contact` 负责奖励最后的接触动作——三个组件形成清晰的 curriculum 链条。

保留 `contact_bonus` 系数 2.0，最大单步奖励从 3.5 降至 2.0，且仅在 `contact_frac > 0` 时激活。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v4。
    
    核心改动：移除 landing_contact 的 base_bonus（持续hover奖励），
    仅在实际有支撑腿接触时给予奖励。解决 agent 在目标附近徘徊而不完成着陆的 exploit。
    
    设计原则：
    - goal_proximity: 二次距离代价 → 引导接近（远处强、近处弱）
    - settling_stability: 二次速度代价 → 引导减速（高速强、低速弱）
    - landing_contact: 仅接触奖励 → 推动完成最后一步
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
    
    # ========== 信号提取（使用 next_obs 评估转移后状态） ==========
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
    
    # ========== Component D: landing_contact — 仅接触奖励 ==========
    # v4 核心改动：移除 base_bonus，仅在实际有腿接触时发放奖励。
    # 这防止了 agent 在"靠近+低速"区域内无限徘徊领取 hover reward。
    # goal_proximity 负责引导接近，settling_stability 负责引导减速，
    # landing_contact 只负责奖励最后的接触动作——三者形成清晰链条。
    near_pad = distance < 0.5
    slow_enough = speed < 1.0
    contact_frac = (left_contact + right_contact) / 2.0  # 0.0 ~ 1.0
    
    if near_pad and slow_enough and contact_frac > 0.0:
        # 连续缩放因子：越近越好、越慢越好
        prox_factor = 1.0 - distance / 0.5      # 1.0@d=0, 0.0@d=0.5
        speed_factor = 1.0 - speed / 1.0        # 1.0@s=0, 0.0@s=1.0
        # 奖励与接触程度成正比
        landing_contact = 2.0 * prox_factor * speed_factor * contact_frac
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