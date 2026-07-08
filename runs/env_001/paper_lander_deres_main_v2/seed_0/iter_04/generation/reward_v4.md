## 诊断

### 1. Evidence

| 信号 | 观察 |
|---|---|
| score | -109.58，全部 20 个 episode 均为 early_terminal（<150 步，score<-50），ep_len 仅 68.45 |
| stability_penalty | magnitude_share 86.9%，signed_share -86.9%，每步约 -0.138，全程激活 |
| progress_delta | signed_share 10.4%，每步约 +0.016，始终被 penalty 淹没 |
| soft_landing_proxy | active_rate 仅 0.9%，从未实质性触发 |
| 历史 iter 2 | 曾尝试不同策略导致 score 暴跌至 -566.8，随后回到 -109.6 |

**行为解读：** agent 在远离目标时就持续受到 velocity/angle/angvel 惩罚。要朝目标移动必然有速度，但有速度就被罚。这两个信号在全局范围互相矛盾，agent 无法学到有效接近策略，约 68 步后坠毁。

### 2. Diagnosis Dimension

**global_to_local_gate** — stability_penalty 作为全局约束，在任务早期（远离目标、需要大动作接近）与 progress_delta 形成直接冲突，导致 agent 无法区分"必要的运动"和"着陆时的晃动"。

### 3. Selected Transformation

**global_to_local_gate：** 将 stability_penalty 乘以一个距离门控因子 `max(0, 1 - dist/gate_radius)`。远离目标时 gate≈0，不施加精细姿态约束；靠近目标时 gate→1，稳定性要求逐步增强。

匹配理由：当前 evidence 完全符合该变换的症状——全局约束在任务无关阶段持续干扰探索，episode 短且以 crash 结束。

### 4. Falsifiable Hypothesis

门控后，agent 在远离目标时只受 progress_delta 引导（无矛盾信号），可以自由学习接近策略；进入门控区后 stability_penalty 逐步激活，自然引导减速和姿态调整。这应使 agent 能存活更久并到达目标附近。

### 5. Expected Next Round

- **score**：显著改善（负值减小），因为 agent 不再因"必要的接近运动"被持续惩罚
- **episode_length**：应增加（不再快速 crash，能到达目标区域）
- **stability_penalty magnitude_share**：应从 86.9% 显著下降，因为大量远离目标的步数不再贡献 penalty
- **soft_landing_proxy active_rate**：应上升，因为 agent 更可能到达着陆区
- **early_terminal**：比例应下降

### 6. Main Risk

如果 gate_radius（1.5）设置不当，agent 可能在门控边界处出现"刹车过晚"——进入门控区时速度仍然过高，来不及减速就 crash。下一轮需观察 episode_length 是否增加但仍在目标附近 crash。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前观测与下一观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 主学习信号：progress delta ----------
    def dist(px, py):
        return (px**2 + py**2)**0.5

    prev_dist = dist(x, y)
    next_dist = dist(next_x, next_y)
    progress_delta = prev_dist - next_dist    # 正值 = 靠近目标

    # ---------- 距离门控：仅在靠近目标时施加稳定性约束 ----------
    gate_radius = 1.5
    distance_gate = max(0.0, 1.0 - next_dist / gate_radius)

    # ---------- 稳定/安全约束：stability penalty（门控后）----------
    speed_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    angvel_penalty = abs(angular_vel)
    stability_penalty_raw = -0.1 * speed_penalty - 0.1 * angle_penalty - 0.05 * angvel_penalty
    stability_penalty = stability_penalty_raw * distance_gate

    # ---------- 稠密着陆引导：continuous soft landing proxy ----------
    # 距离因子：越靠近目标越好
    proximity_factor = max(0.0, 1.0 - next_dist / 1.0)
    # 速度因子：越慢越好
    speed = (vx**2 + vy**2)**0.5
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    # 姿态因子：越正越好
    upright_factor = max(0.0, 1.0 - abs(angle) / 0.5)
    # 接触因子：双腿着地更好
    contact_factor = (left_contact + right_contact) / 2.0
    # 几何平均防止单一因子优秀时掩盖其他因子
    core_quality = (proximity_factor * speed_factor * upright_factor) ** (1.0 / 3.0)
    soft_landing_proxy = 0.5 * core_quality * (0.3 + 0.7 * contact_factor)

    # ---------- 合成 ----------
    total_reward = progress_delta + stability_penalty + soft_landing_proxy

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'soft_landing_proxy': soft_landing_proxy,
    }
    return float(total_reward), components
```