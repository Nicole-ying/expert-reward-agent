# 设计理由

## 为什么以前都失败了
回顾全部 6 轮迭代：

- **Iter1** 简单 `velocity_gated` 就跑出了 **294.82**，之后任何改动（加重稳定性惩罚、加 action penalty、加 stability bonus）都没有再超越它。
- **Iter2–6** 试图通过“加大独立稳定性信号”来突破瓶颈，但每次都 **要么压制速度（Iter2/3），要么信号冗余（Iter6 同时有两个乘性因子叠加在速度上），导致梯度混杂、策略未优于简洁基线。**
- 根本问题：**主信号本身太弱**（线性速度奖励），策略停留在“足够安全的速度平台”，不再追求更高速度。新增的 stability 组件没有创造新速度潜能，只是改变了安全边际，干扰了原始学习。

## 新骨架选了什么算子、与已尝试过的本质不同
- **过去骨架**：`velocity_gated + stability_penalty/bonus + action_efficiency` — 主要把稳定性当作独立加法/乘法组件，导致多信号竞争。
- **新骨架**：**仅保留一个主信号 `velocity_squared_gated`，使用非线性、陡峭衰减的健康门控（`soft_health_gate`），把“保持直立”完全内化为主奖励的乘数。**
- **速度平方（`dense_state_signal` 凸化）**：突破线性平台，给予更高的速度以指数级更强的梯度，逼迫策略进化出更快步态。
- **陡峭非线性 gate**（`1/(1+80*excess²)`）：在 safe_angle 0.25 rad 以内 gate≈1；一旦倾角变大，奖励急剧衰减——确保“高速但危险”的瞬时期望奖励 **低于** “中速且安全”，从而使速度最大化与防摔倒**自然统一**，不必另加 stability bonus。
- **极微动作惩罚**（`quadratic_penalty`，系数 0.005，为 Iter4 的 1/4）：提供微弱能量效率信号，但量级远小于主信号，**不会压制速度探索**。

这一骨架与 Iter1 的哲学一致（简洁、无冗余组件），但把原来的线性速度用平方和更陡的 gate 替换，直指当前瓶颈：**速度梯度不足**。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取关键观测
    hull_angle_next = next_obs[0]
    horizontal_velocity_next = next_obs[2]

    # ---------- 非线性健康门控 ----------
    # 倾角在 0.25 rad 内几乎不衰减，超过后迅速塌缩
    safe_angle = 0.25
    excess = max(0.0, abs(hull_angle_next) - safe_angle)
    gate = 1.0 / (1.0 + 80.0 * excess ** 2)

    # ---------- 主学习信号：速度平方 × 门控 ----------
    # 速度平方提供指数级梯度，推动突破线速度平台
    velocity_squared_gated = gate * horizontal_velocity_next ** 2

    # ---------- 极微能量效率信号 ----------
    # 系数极小，仅暗示节能，绝不压抑探索
    action_penalty = -0.005 * (
        action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2
    )

    # ---------- 总奖励 ----------
    total_reward = velocity_squared_gated + action_penalty
    components = {
        "velocity_squared_gated": velocity_squared_gated,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺高速推进的强梯度；过去冗余的稳定性组件可合并进乘性 gate。
- **behavior**: agent 在安全速度平台停滞，始终未能刷新 Iter1 的 294.82。
- **signal**: 主信号需从线性过渡到凸化，同时用陡峭 gate 防止‘高速+倾斜’策略作弊。
- **level**: Level 3 (重建骨架)
- **hypothesis**: 用 `v²` 提供突破性速度梯度；陡峭非线性 gate 使稳定性内化为主乘数，不再需要独立 stability 组件，避免历史冗余，从而超越 294.82 并向 300 逼近。
- **risk**: gate 过于陡峭可能导致早期探索过慢（agent 过于不敢倾侧）；若发生，需在后续微调 safe_angle 或衰减系数。