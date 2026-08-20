# Response Record

# 设计理由
当前奖励函数已使智能体稳定超额完成目标（score=319.1，terminated=20/20 success）。然而 `tilt_penalty` 与 `health gate` 功能重叠：二者都是对 `hull_angle` 的惩罚，前者是独立二次惩罚，后者已将倾斜映射为前进奖励的折扣因子，提供了同样的梯度信息。  

移除 `tilt_penalty` 可以：
1. **消除冗余**：`health gate` 已经让倾斜增大时 `forward_progress` 缩水，额外惩罚不再提供独立信号。
2. **释放步态自然度**：正常行走中躯干存在小幅摆动，二次惩罚会不加区分地抑制这些微小倾斜，移除后智能体可更自由地寻找高效步态。
3. **简化奖励结构**：与 iter 4 的简化方向一致，进一步剥离多余项，降低调参风险。

其他组件保持不变，尺度无需调整。移除后主信号仍为前进速度，动作成本依然保留以鼓励节能，整体结构更清晰。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Main learning signal: forward velocity, gated by posture health
    forward_vel = next_obs[2]                 # positive = moving forward
    hull_angle = next_obs[0]                  # radians, 0 = vertical
    health = 1.0 / (1.0 + 10.0 * (hull_angle ** 2))   # posture gate
    forward_progress = 1.0 * forward_vel * health

    # Efficiency: light action cost
    avg_action_power = (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2) / 4.0
    action_cost = -0.05 * avg_action_power

    total_reward = forward_progress + action_cost

    components = {
        'forward_progress': forward_progress,
        'action_cost': action_cost
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 奖励覆盖了进展信号与摔倒前兆，仅 `tilt_penalty` 与 `health gate` 功能完全重叠，属于冗余惩罚。
- **behavior**: 智能体已稳定完成地形，步态高效且无摔倒终止。
- **signal**: 缺少独立的跳跃抑制信号（但当前表现已合格），冗余倾斜惩罚无必要地约束了正常摆动。
- **level**: Level 2（结构变换，移除冗余组件）
- **hypothesis**: 移除冗余倾斜惩罚后，智能体在无额外干扰下仍可保持直立，且步态自然度略升，性能持平或微增。
- **risk**: 极低。若未来训练中倾斜失控，`health gate` 仍会提供足够梯度惩罚。
