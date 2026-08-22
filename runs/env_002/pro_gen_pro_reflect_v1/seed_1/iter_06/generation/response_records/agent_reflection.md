# Response Record

# 设计理由

## 审计发现
- 终止前兆缺失：身体摔倒终止，但当前代码只用角速度hinge做稳定性惩罚，该惩罚active_rate=0%，完全失效。
- 目标进度断裂：velocity_gated只奖励"当前速度"，没有直接奖励"走多远"的累积进度。
- 效率信号已有：action_efficiency存在且适度，非瓶颈。
- 僵尸组件：stability_penalty active_rate=0%，必须改造。

## 行为诊断
Agent当前len=1113.7、score=287.7，行走距离足够但速度被gate压制。gate在0.8-1.2弧度间线性变化，agent为维持高gate被迫保持小角度，但这牺牲了激进步态和前向速度。稳定性惩罚阈值2.0过高（正常行走角速度很少超过此值），导致active_rate为0%，完全无梯度提供。agent处于"安全但慢"的局部最优，需要更精确的稳定性信号来突破。

## 干预目标
改造稳定性组件：从失效的角速度hinge → 基于身体角度的progress_ratio形态。让agent在维持安全姿态的同时获得更多探索自由，促进速度提升。

## 数学形式
新组件 `stability_ratio`：
- 核心思想：身体角度偏离竖直越多，稳定性折扣越大
- 安全区（abs(hull_angle) < 0.15）：ratio ≈ 1.0
- 危险区（0.15 → 0.8）：sigmoid式下降
- 致命区（> 0.8）：ratio ≈ 0，但gate会严重压制，joint condition持续提供梯度
- 公式：`1.0 / (1.0 + 25.0 * max(0, abs(hull_angle) - 0.15)**2)`
- 权重0.5：基于episode_sum_mean/len ≈ 0.27的主信号per-step，新惩罚目标per-step约0.05-0.10，为主信号的0.2-0.4x

## 为什么这能改善
原stability_penalty threshold 2.0切断了所有梯度；新ratio在正常行走角度范围[0, 0.5]内持续提供光滑梯度，引导agent维持更直立姿态的同时允许适度动态摆动，解放前向速度。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    hull_angle_next = next_obs[0]
    hull_ang_vel_next = next_obs[1]
    horizontal_velocity_next = next_obs[2]

    # ---------- 健康门控 ----------
    danger_angle = 0.8
    max_angle = 1.2
    gate = max(0.0, min(1.0, (max_angle - abs(hull_angle_next)) / (max_angle - danger_angle)))

    # ---------- 主学习信号 ----------
    velocity_gated = gate * horizontal_velocity_next

    # ---------- 姿态稳定性：连续折扣 ----------
    safe_offset = 0.15
    excess_angle = max(0.0, abs(hull_angle_next) - safe_offset)
    stability_ratio = 1.0 / (1.0 + 25.0 * excess_angle ** 2)  # 安全区≈1, 危险区快速衰减
    stability_bonus = 0.5 * stability_ratio * horizontal_velocity_next

    # ---------- 动作效率 ----------
    w_action = 0.02
    action_efficiency = -w_action * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # ---------- 总奖励 ----------
    total_reward = velocity_gated + stability_bonus + action_efficiency
    components = {
        "velocity_gated": velocity_gated,
        "stability_bonus": stability_bonus,
        "action_efficiency": action_efficiency
    }
    return float(total_reward), components
```

# 诊断摘要

- **audit**: 缺少姿态稳定性与前进进度的直接耦合；旧稳定性惩罚阈值过高导致失效
- **behavior**: agent处于"安全但慢"的局部最优，gate压制速度，无梯度突破
- **signal**: 稳定性梯度缺失（active_rate 0%），需要连续姿态信号
- **level**: Level 2
- **hypothesis**: 用姿态稳定性折扣乘以前向速度，创建"安全快走>安全慢走>不安全>摔倒"的梯度排序，引导agent在维持安全前提下提高速度
- **risk**: stability_bonus可能过度奖励低速下的稳定站立，但gate对极端角度已有限制，且horizontal_velocity因子确保只有前进时才有bonus
