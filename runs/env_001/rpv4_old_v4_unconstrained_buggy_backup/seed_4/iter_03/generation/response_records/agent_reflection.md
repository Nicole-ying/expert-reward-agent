# Response Record

现在我对所有数据有了完整理解。让我综合诊断。

---

## 诊断报告

### Evidence

| 证据 | 数据 | 含义 |
|---|---|---|
| `goal_proximity` 跨 iter_1→iter_2 | +2.149→-33.25，magnitude_share 29.6%→61.5% | 从 improvement_delta 切到 state-cost 后，距离惩罚爆炸式增长，成为主导信号 |
| `goal_proximity` checkpoint 趋势 | cp10%–cp100% 全部 ≈ -33.25，完全平坦 | **死梯度**：线性惩罚梯度恒为 -0.5，agent 无法区分好坏动作 |
| `settling_stability` checkpoint 趋势 | cp10%–cp100% 全部 ≈ -18.9，完全平坦 | **死梯度**：同因，速度惩罚没有距离/速度相关的梯度变化 |
| `attitude_constraint` checkpoint 趋势 | -2.023→-0.702，持续改善 | agent **确实在学习**姿态控制，这是有效的 |
| `landing_contact` checkpoint 趋势 | 0.79→1.165，持续改善 | agent **确实在学习**着陆接触 |
| `goal_proximity` + `settling_stability` | 合计 magnitude_share = 96.4% | 两个死梯度组件占据了几乎全部 reward 量级，淹没了有效学习信号 |
| `landing_contact` active_rate | 仅 0.7% | 门控太严格（distance<0.3, speed<0.6），agent 很少触发正向反馈 |
| 20/20 全部提前终止 | len=68.35，score=-113.6 | agent 在约 68 步时 crash，从未真正完成着陆 |
| iter_1→iter_2 score | -121.5→-113.6 (+7.9) | 有改善但来自 attitude 和 landing，导航本身无进展 |

### Behavior Diagnosis

当前策略的行为模式：agent 在下落过程中学会了保持直立姿态（attitude_constraint 持续改善），并且在接近垫板时尝试接触（landing_contact 改善）。但 **agent 没有学会主动导航到目标垫板**——goal_proximity 和 settling_stability 的平坦趋势表明，无论训练进行多久，agent 的距离和速度分布纹丝不动。这是因为：

1. **线性惩罚 `-0.5*distance` 的梯度恒为常数**：无论 agent 距离目标 3 单位还是 0.3 单位，每步动作对 reward 的影响都是 `-0.5 * Δdistance`，而 Δdistance 通常极小（0.01-0.05），RL 无法在噪声中检测到如此微弱的梯度。
2. **这两个死梯度组件 magnitude_share 合计 96.4%**，将 landing_contact 的正向信号完全淹没——即使 agent 完美着陆获得 +2.5，也仅抵消 5 步距离惩罚。

**核心问题**：state-cost 框架方向正确，但线性形式提供了恒定弱梯度。需要改为梯度随状态变化的非线性形式。

### 修改方案（L2 — 数学形态变更）

1. **goal_proximity**：线性 `-0.5*d` → 二次 `-0.1*d²`。梯度 = `-0.2*d`，距离越远梯度越强，形成自然 curriculum。
2. **settling_stability**：线性 `-0.6*s*gate` → 二次 `-0.3*s²*gate`。梯度 = `-0.6*s*gate`，速度越高梯度越强。
3. **landing_contact**：硬门控 → 渐进式奖励。扩展门控范围（distance<0.5, speed<1.0），使用连续缩放因子，加入 base_bonus（靠近+低速即有奖励）+ contact_bonus（接触额外奖励），提高 active_rate。
4. **attitude_constraint**：保持结构但提高远距离时的下限权重（0.15→0.3），让姿态学习更早开始。
5. 整体 magnitude 重新平衡，确保 landing_contact 的正向信号不被淹没。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v3。
    
    核心改动：从线性 state-cost → 二次 state-cost（梯度随状态变化）
    - goal_proximity: 二次距离惩罚，梯度 ∝ distance（远处强、近处弱）
    - settling_stability: 二次速度惩罚，梯度 ∝ speed（高速强、低速弱）
    - landing_contact: 渐进式奖励，软化门控提高 active_rate
    - attitude_constraint: 保持结构，提高远距离基线权重
    
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
    # 梯度 = -0.2 * distance，远处梯度强（d=3→-0.6），近处梯度弱（d=0.3→-0.06）
    # 天然 curriculum：远处强力驱动靠近，近处精细调控
    goal_proximity = -0.1 * distance ** 2
    
    # ========== Component B: settling_stability — 二次速度代价 ==========
    # 梯度 = -0.6 * speed * gate，高速时梯度强，且随接近目标增强
    # gate 范围：远距离约 0.28，近距离约 1.0，提供 ~3.6x 动态范围
    settling_stability = -0.3 * speed ** 2 * (0.2 + 0.8 * proximity_gate)
    # 角速度惩罚（新增独立项）
    settling_stability += -0.08 * ang_vel ** 2 * (0.2 + 0.8 * proximity_gate)
    
    # ========== Component C: attitude_constraint — 姿态代价 ==========
    # 保持线性但将远距离基线从 0.15 提高到 0.3，更早开始姿态学习
    angle_error = abs(body_angle)
    ang_vel_mag = abs(ang_vel)
    attitude_gate = 0.3 + 0.7 * proximity_gate
    attitude_constraint = -0.4 * angle_error * attitude_gate
    attitude_constraint += -0.15 * ang_vel_mag * attitude_gate
    
    # ========== Component D: landing_contact — 渐进式着陆奖励 ==========
    # 软化门控：distance<0.5（原 0.3），speed<1.0（原 0.6）
    # 使用连续缩放因子替代二元门控，提供稠密梯度
    near_pad = distance < 0.5
    slow_enough = speed < 1.0
    
    if near_pad and slow_enough:
        # 连续因子：越近越好、越慢越好
        prox_factor = 1.0 - distance / 0.5      # 1.0@d=0, 0.0@d=0.5
        speed_factor = 1.0 - speed / 1.0        # 1.0@s=0, 0.0@s=1.0
        contact_frac = (left_contact + right_contact) / 2.0  # 0.0~1.0
        
        # base_bonus: 靠近+低速即有奖励，引导 agent 先学会接近
        base_bonus = 1.5 * prox_factor * speed_factor
        # contact_bonus: 接触额外奖励，推动完成最后一步
        contact_bonus = 2.0 * prox_factor * speed_factor * contact_frac
        
        landing_contact = base_bonus + contact_bonus
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
