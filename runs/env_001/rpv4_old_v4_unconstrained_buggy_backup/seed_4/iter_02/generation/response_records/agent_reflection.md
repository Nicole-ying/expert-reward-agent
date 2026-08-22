# Response Record

现在我对数据有了完整的了解。让我来分析并给出诊断和修改方案。

---

## 诊断报告

### Evidence

| 证据 | 数据 | 含义 |
|---|---|---|
| `goal_proximity` 在所有 checkpoint 完全平坦 | cp10%–cp100% 全部 ≈ 2.149 | **死梯度**：agent 没有学习改进位置，reward 来自重力自然下落 |
| `settling_stability` 持续下降 | cp10%=0.563 → cp100%=0.445 | agent 学到的不是减速，而是**加速**（更快的自由落体） |
| score 全程无改善 | -113.9 ⇄ -121.3，震荡无趋势 | RL 没有学到任何有效策略 |
| 20/20 集全部提前终止 | len=68.5，score<-50 占 19/20 | 全部坠落/碰撞，无一成功着陆 |
| `attitude_constraint` 全程为负且 100% 激活 | ep_sum≈-1.8 | 远距离时 gate=0.1，但姿态在坠落中不断恶化，形成持续拖累 |
| `landing_contact` 几乎死亡 | active_rate=0.2%，仅在偶然触发时给 0.5 | 门控过严，agent 从未真正到达着陆状态 |

### Behavior Diagnosis

当前奖励函数的核心问题：**`improvement_delta`（`current_distance - next_distance`）在自由落体中天然为正**。重力替 agent 完成了"接近目标"的工作，agent 无需学习任何引擎控制就能持续获得 `goal_proximity` 正向奖励。`settling_stability` 的 `speed_improvement` 权重（0.5×proximity_gate≈0.12 远距离时）远不足以抵消这个"免费奖励"。

结果：agent 学到的策略是**什么都不做，让重力带着坠落，快速触发 crash 终止**——因为任何引擎使用都会触发 `fuel_efficiency` 惩罚，而坠落本身不需要付出任何代价就能获得距离改善奖励。这是一个典型的 **improvement_delta 被环境物理劫持**的失效模式。

### 修改方案

**L2 修改**：将 `goal_proximity` + `settling_stability` 从 improvement_delta（改进量奖励）切换为 **state-cost（状态代价）** 框架。

核心思路：
- **不再奖励"距离减少"**（重力会免费提供），改为**惩罚"距离大"**（agent 必须主动靠近才能消除代价）
- **不再奖励"速度降低"**，改为**惩罚"速度高"**，且惩罚随接近目标而指数增强
- 着陆奖励分层：双接触 > 单接触 > 无接触，软化门控条件提高 active_rate
- 姿态约束保持 proximity-gated 结构，但改用状态代价而非改进量

预期效果：自由落体每步积累约 -1.5 到 -2.0 的代价，而成功着陆后每步净收益约 +1.5 到 +2.0。agent 必须学会用引擎控制下降才能终止代价积累。

---

## 新奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v2。
    
    核心改动：从 improvement_delta（改进量）框架切换到 state-cost（状态代价）框架。
    - 不再奖励"距离减小"（重力会免费提供），改为惩罚"距离大"
    - 不再奖励"速度降低"，改为惩罚"速度高"，且随接近目标增强
    - 着陆奖励分层，软化门控提高 active_rate
    
    Args:
        obs: 当前观测 (8,)
        action: 当前动作 (0-3)
        next_obs: 下一观测 (8,)
        original_reward: 未使用
        info: 未使用
        training_progress: 未使用
    """
    
    # ========== 信号提取（主要使用 next_obs 评估转移后的状态） ==========
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
    
    # proximity_gate: 1.0 在目标处，~0.1 在远处 (distance≈3)
    proximity_gate = 1.0 / (1.0 + 3.0 * distance)
    
    # ========== Component A: goal_proximity — 距离代价 ==========
    # 距离越远代价越大，驱使 agent 靠近目标垫板
    # 线性代价，在 distance≈3 时约 -1.5，在 distance≈0 时约 0
    goal_proximity = -0.5 * distance
    
    # ========== Component B: settling_stability — 速度代价 ==========
    # 速度越高代价越大，且随接近目标急剧增强
    # 远距离时弱（允许快速接近），近距离时强（强制减速）
    settling_stability = -0.6 * speed * (0.15 + 0.85 * proximity_gate)
    
    # ========== Component C: attitude_constraint — 姿态代价 ==========
    # 角度和角速度偏离的代价，同样随 proximity 增强
    angle_error = abs(body_angle)
    ang_vel_mag = abs(ang_vel)
    attitude_constraint = -0.4 * angle_error * (0.15 + 0.85 * proximity_gate)
    attitude_constraint += -0.15 * ang_vel_mag * (0.15 + 0.85 * proximity_gate)
    
    # ========== Component D: landing_contact — 着陆奖励 ==========
    # 分层奖励：双接触+直立 > 单接触 > 靠近但无接触
    near_pad = distance < 0.3
    slow_enough = speed < 0.6
    upright = angle_error < 0.3
    dual_contact = (left_contact > 0.5 and right_contact > 0.5)
    any_contact = (left_contact > 0.5 or right_contact > 0.5)
    
    if near_pad and slow_enough:
        if dual_contact and upright:
            landing_contact = 2.5  # 完美软着陆
        elif dual_contact:
            landing_contact = 1.5  # 双接触但姿态不佳
        elif any_contact:
            landing_contact = 0.8  # 单侧接触
        else:
            landing_contact = 0.2  # 靠近+低速但未接触（鼓励继续下降至接触）
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
