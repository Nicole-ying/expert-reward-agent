# Response Record

# 设计理由

**为什么以前都失败了？**  
回顾累积记录，所有超出初始骨架的修改（iter2–iter7）都试图通过加强惩罚、门控或速度凸化来精调策略，但每一个都在探索阶段破坏了原始骨架中有效的梯度结构。  
- iter1 的简单配方（速度正奖励 + 轻量稳定性惩罚）给出了最佳 294.82，说明 agent 能学会稳健快走。  
- iter2 加大稳定性惩罚后 len 和 score 一起下降，因为惩罚挤占了速度的激励。  
- iter3 引入动作惩罚后 score 断崖式崩溃，惩罚过重使 agent 不敢动弹。  
- iter4 虽降低惩罚系数拉回了 len，但 score 仍低于 best。  
- iter5 的 hinge 姿态惩罚使 len 超长但速度停滞，证明独立惩罚无法引导加速。  
- iter6 的稳定性折扣速度再次导致 score 衰退。  
- iter7 使用速度平方 + 极端陡峭门控 (`1/(1+80·excess²)`) 导致 score 从 276 暴跌到 195：门控在 hull_angle > 0.25 rad 后迅速塌缩，把安全区域压缩得太窄，agent 为了避免门控塌缩而极度保守，速度上不去。  

连续 ≥3 轮 ❌（iter2、3、6、7）表明同一方向的修补（加约束 / 凸化速度）在治标。  
**核心病因**：任何使“高速行为”与“安全行为”严重冲突的机制（过陡门控、过强惩罚、平方加速）都会逼迫 agent 放弃速度去换取安全，无法接近 target 300。

**新骨架的逻辑**  
采用 **线性速度主奖励 × 平缓姿态门控** + **极轻动作惩罚** 的复合骨架，重建一条平滑且易探索的梯度路径：
- 主信号 `horizontal_velocity` 使用线性形式，提供稳定的正向梯度，不因凸化放大波动。
- 门控 `1 / (1 + (|hull_angle| / 0.5)²)` 在正常姿态摆动（~0.1‑0.3 rad）内保持 0.8‑0.96，只在接近危险倾斜时才明显衰减。这比 iter7 的陡峭门宽得多，让 agent 有胆量提高速度。
- 动作惩罚 `-0.02·∑action²` 与 iter4 的轻量系数一致，提供能量效率暗示但不压制动作幅度。
- 额外加入 `vertical_velocity` 平方惩罚以抑制无效的上下颠簸，平滑步态，系数极小（0.05），不影响主信号。

**与已尝试过的本质区别**：
- 不同于 iter1 的 `stability_penalty + velocity_gated`，本骨架用 **乘积式门控** 替代独立惩罚，避免安全感妨碍速度。
- 不同于 iter7，本骨架用 **线性速度** 替代速度平方，**平缓门控** 替代急塌门控，确保高速探索不被惩罚性切断。
- 整体奖励曲面更连续、梯度更平坦，适合在 iter1 的平台期上进一步推动速度。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 姿态门控 ----------
    # 使用平方反比形式，hull_angle=0.5 时 gate≈0.5，0.25 时≈0.8，
    # 确保正常摆动不会过分衰减速度奖励。
    hull_angle_next = next_obs[0]
    gate = 1.0 / (1.0 + (abs(hull_angle_next) / 0.5) ** 2)

    # ---------- 主学习信号：线性速度 × 门控 ----------
    horizontal_velocity_next = next_obs[2]
    velocity_gated = gate * horizontal_velocity_next

    # ---------- 轻量动作效率信号 ----------
    # 每步轻微惩罚力矩平方和，促使策略寻找节能步态。
    action_penalty = -0.02 * (action[0] ** 2 + action[1] ** 2 + 
                              action[2] ** 2 + action[3] ** 2)

    # ---------- 极轻微垂直速度惩罚 ----------
    # 抑制上下颠簸，帮助步态平滑。
    vertical_velocity_next = next_obs[3]
    vertical_penalty = -0.05 * vertical_velocity_next ** 2

    # ---------- 总奖励 ----------
    total_reward = velocity_gated + action_penalty + vertical_penalty
    components = {
        "velocity_gated": velocity_gated,
        "action_penalty": action_penalty,
        "vertical_penalty": vertical_penalty
    }
    return float(total_reward), components
```

---

# 诊断摘要
- **audit**: 上一轮奖励缺少对前进速度的稳定激励，且门控过陡抑制高速探索；当前重构补全了线性速度门控、能效和垂直抑制。
- **behavior**: agent 在 iter7 中因陡峭门控而高度保守，速度提不上去，score 停滞在 195。
- **signal**: 需要恢复速度的正向梯度并放宽安全姿态范围。
- **level**: Level 3（重建骨架，更换信号组合与门控形态）
- **hypothesis**: 平缓衰减的门控与线性速度奖励的组合，能在鼓励快速前进的同时避免惩罚性崩溃，使 agent 突破 294 的平台向 300 靠近。
- **risk**: 如果垂直速度惩罚系数不当，可能轻微压制跳跃式步态，影响极限速度；可通过后续调度微调。
