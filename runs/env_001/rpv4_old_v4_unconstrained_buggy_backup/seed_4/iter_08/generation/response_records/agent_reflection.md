# Response Record

---

## 诊断报告

### Evidence

| 证据 | 数据 | 含义 |
|---|---|---|
| **iter_7 score 已超目标** | score=233.50 > target=200, terminated=19/20 | 任务已解决，进入精调阶段 |
| **landing_contact 占比 70%** | ep_sum=78.34, magnitude_share=70.0%（iter_4 仅 41%） | 接触奖励主导了整个 reward 构成，其他组件被边缘化 |
| **checkpoint 剧烈振荡** | landing_contact: cp60%=71.7 → cp70%=6.1 → cp80%=6.8 → cp90%=6.8 → cp100%=54.5 | 中期策略在"大量接触"和"几乎不接触"之间反复横跳，训练不稳定 |
| **score 随之剧烈振荡** | cp60%=132.5 → cp70%=46.2 → cp80%=19.5 → cp90%=25.3 → cp100%=242.8 | landing_contact 主导导致 score 方差极大 |
| **iter_7 vs iter_4 对比** | 系数 2.0→1.2 后 landing_contact 反而从 35→78 暴涨 | 降低系数并没有削弱接触信号——agent 学会了更频繁/更优地触发它，说明信号仍然过强 |
| **score range 宽** | [153.93, 277.80]，最低分仍低于 200 | 20 集中有 episode 未达目标，表现不够一致 |
| **历史教训** | iter_3→iter_4: base_bonus 移除后 score -23→155；iter_5→iter_6: 势能差修改导致崩溃 | 任何对 landing_contact 的结构性放大都会 exploit；任何结构性重写都可能崩溃 |

### Behavior Diagnosis

当前策略已学会有效着陆（19/20 terminated），且比 iter_4 更快（len 446 vs 515）。但训练过程极不稳定：agent 在 cp60% 发现大量接触奖励（ep_sum=71.7, active=28.3%），随后 cp70-90% 策略崩溃（active 降至 1-2%），最后 cp100% 再次恢复。这说明 **landing_contact 的 70% magnitude share 使得策略对一个组件的波动过度敏感**——当接触行为稍有退化，总分断崖式下跌（cp70%=46），导致后续探索困难。

此外，penalty 组件（goal_proximity=-10.9, settling=-5.3, attitude=-8.0, fuel=-9.3）的总 magnitude 仅 ~33.6，而 landing_contact 单组件达 78.3，比例为 **2.3:1**。这个不平衡意味着 agent 的主要优化方向是「最大化接触奖励」而非「最小化综合代价」。

### 干预方案

**L1 — 继续降低 landing_contact 系数：1.2 → 0.9**

- **理由**：iter_7 将系数从 2.0 降至 1.2 效果显著（score 155→233），但 70% share 表明力度仍不足。进一步降至 0.9（~25% reduction）将使 landing_contact 的 magnitude 降至 ~58，占比降至 ~63%，penalty 组件获得更多相对权重，训练曲线应更平滑。
- **为什么不做 L2**：iter_7 的 v4 结构已被验证有效，且历史上 iter_5/iter_6 的 L2 修改（transition reward、势能差）均导致崩溃。在已超目标的情况下，保守 L1 是最安全的选择。
- **预期**：score 可能微降至 200-215 区间，但 training stability 改善，score range 收窄。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v8。

    核心改动（L1）：landing_contact 系数 1.2 → 0.9，进一步降低接触奖励的
    magnitude share（v7=70%），使 penalty 组件获得更合理的相对权重，
    减少训练中期的策略振荡，提升最终策略的一致性。

    设计原则（继承 v4/v7）：
    - goal_proximity: 二次距离代价 → 引导接近（远处强、近处弱）
    - settling_stability: 二次速度代价 → 引导减速（高速强、低速弱）
    - landing_contact: 连续接触奖励 → 奖励"接近+慢速+接触"的着陆过程
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

    # ========== Component D: landing_contact — 连续接触奖励 ==========
    # v8：系数 1.2 → 0.9（L1 微调），降低 magnitude share 以减少训练振荡。
    # 仅在实际有支撑腿接触、靠近垫板、且速度足够低时发放奖励。
    # 三个条件共同防止 agent 在无关阶段获得误奖励：
    #   - near_pad: 距离 < 0.5，确保只在目标附近激活
    #   - slow_enough: 速度 < 1.0，防止高速撞击时领奖
    #   - contact_frac > 0: 至少一条腿接触，推动完成最后一步
    near_pad = distance < 0.5
    slow_enough = speed < 1.0
    contact_frac = (left_contact + right_contact) / 2.0  # 0.0 ~ 1.0

    if near_pad and slow_enough and contact_frac > 0.0:
        # 连续缩放因子：越近越好、越慢越好
        prox_factor = 1.0 - distance / 0.5      # 1.0@d=0, 0.0@d=0.5
        speed_factor = 1.0 - speed / 1.0        # 1.0@s=0, 0.0@s=1.0
        # v8: 系数 0.9（v7=1.2, v4=2.0），平衡 reward composition
        landing_contact = 0.9 * prox_factor * speed_factor * contact_frac
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
