# Response Record

1. `evidence`：历史最佳（iter 2）骨架为 approach_reward + stable_bonus + fuel_penalty + goal_progress，得分 194.87，后续所有骨架均远低于此且多以快速失败告终；iter 8 的 landing_proxy 激活率 0.5%、x_penalty 激活率 0%，说明复杂组件未提供有效梯度；当前 REBUILD MODE 要求避开已证明失败的路径，回归简单骨架。

2. `behavior_diagnosis`：所有 episode 均以 early-terminal 结束（len ≈ 87），策略尚未学会受控下降与着陆，主要因为缺乏持续、可感知的“成功状态”引导信号，agent 仅依赖距离缩短奖励导致危险的高速坠落。

3. `signal_completeness`：成功着陆所需的低距离、低速度、小姿态角、双腿接触均可从观测连续构造，构成密集的成功 proxy 梯度；燃料效率可通过离散动作惩罚表达；灾难性出界可通过 x 边界 hinge 补充但当前未严重激活，暂不纳入。

4. `selected_level`：Level 3 — 基于历史最优骨架族重建，避开后续复杂角度/速度惩罚和多组件过载的失败路径。

5. `selected_intervention`：新骨架仅包含三个组件：`approach_delta`（距离缩短，系数 8.0）、`success_proxy`（着陆软奖励，系数 3.0）、`fuel_penalty`（离散动作惩罚 -0.05）。

6. `falsifiable_hypothesis`：增加连续化的 `success_proxy` 能提供“靠近目标且稳定”的每步正向梯度，迫使 agent 在接近平台时主动减速并对齐姿态，从而终止坠落循环，episode 长度和最终得分将显著提升。

7. `expected_next_round`：len 应＞300（而非当前 87），score 应明显转正并可能接近或超过 0；success_proxy 的 active_rate 将显著高于 iter 8 的 0.5%，approach_delta 保持正值但不腐蚀为加速坠落。

8. `main_risk`：success_proxy 在早期阶段（距离远、速度大）可能恒为 0，导致梯度信号不足，agent 仍依赖 approach_delta 快速下降，此时缺乏速度抑制机制可能延缓学习。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 计算当前和下一步到目标中心的欧氏距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    
    # 1. 距离缩短奖励（靠近目标为正）
    approach_delta = current_dist - next_dist
    approach_reward = 8.0 * approach_delta
    
    # 2. 成功着陆软奖励：联合条件代理（几何平均防塌缩）
    f_dist = max(0.0, 1.0 - next_dist / 0.3)               # 距离因子
    speed_mag = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    f_speed = max(0.0, 1.0 - speed_mag / 0.5)              # 速度因子
    f_angle = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)      # 姿态角因子
    f_contact = (next_obs[6] + next_obs[7]) / 2.0          # 接触因子
    success_proxy = (f_dist * f_speed * f_angle * f_contact) ** 0.25
    success_reward = 3.0 * success_proxy
    
    # 3. 燃料消耗惩罚（离散动作：非零动作即惩罚）
    fuel_penalty = -0.05 if action != 0 else 0.0
    
    total = approach_reward + success_reward + fuel_penalty
    
    components = {
        'approach_delta': float(approach_delta),
        'success_proxy': float(success_proxy),
        'fuel_penalty': float(fuel_penalty)
    }
    return (float(total), components)
```
