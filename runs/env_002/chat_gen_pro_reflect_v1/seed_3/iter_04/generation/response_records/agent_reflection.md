# Response Record

# 设计理由

当前 reward 结构失衡：`gait_quality` 在 magnitude share 中占据 76.5%，而 `forward_progress` 的 active_rate 仅 2.1% 且 episode_sum 几乎为零。agent 在 1600 步中完美存活但 horizontal_velocity ≈0 — 它学到了“原地完美交替踏步”即可稳定获取 `gait_bonus`，**主进度信号完全失活**。  
此外，`balance_penalty` 对 `hull_angle` 的全局二次惩罚（active_rate=100%）抑制了机器人任何微小前倾，而前倾恰好是产生 forward velocity 的必要条件（重力分量驱动）。这使得探索前进的路径被封死，agent 被锁定在直立原地踏步的局部最优。

### 修改了什么组件
**`balance_penalty`：从无界二次惩罚 → hinge + 小幅角速度惩罚**

- **angle 部分**：原 `-0.5 * hull_angle^2` 对零倾角附近也有压制，现在改为 `-w * max(0, |hull_angle| - threshold)^2`，在安全区间内无惩罚，超出才惩罚。
- **angular_vel 部分**：系数从 0.1 降至 0.05，避免过度压制动态调节。
- **理由**：释放机器人前倾探索的空间，使其有机会获得 forward velocity，同时仍对即将摔倒的大角度倾斜保持强约束。

该修改是 **Level 2 结构变换**（全局惩罚→局部门控），因为任务目标与当前惩罚在早期探索阶段存在冲突，需要移除“安全但无进度”区域的压制信号。

### 系数与阈值校准
- `threshold = 0.3`（约 17°）：远在摔倒临界角（估计 ≥ 1.0 rad）的 60‑80% 以内，允许无惩罚的前倾探索。
- `angle_penalty_coef = 1.0`：当倾斜超过 0.3 时恢复较强惩罚，防止快速恶化。
- `angular_vel_coef = 0.05`：约为原 0.1 的一半，保留对异常旋转的软阻尼但不主导。

这些值确保新的 `balance_penalty` per‑step 负担远小于 `gait_bonus`（~0.3），且 `forward_progress` 一旦被激活就能形成有力引导，不会因惩罚过重而抵消。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号（保持原样，等待 balance 释放后激活）
    # ============================================================
    horizontal_vel = obs[2]
    forward_progress = 2.0 * (max(0.0, horizontal_vel) ** 2)

    # ============================================================
    # 2. gait_bonus: 步态交替质量（保持原样）
    # ============================================================
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    gait_quality = 1.0 - abs(leg1_contact - leg2_contact)
    gait_bonus = 0.3 * gait_quality

    # ============================================================
    # 3. balance_maintenance: 改 hinge 版，允许小幅度前倾探索
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    angle_deviation = abs(hull_angle) - 0.3
    angle_penalty = -1.0 * (max(0.0, angle_deviation) ** 2)
    angular_vel_penalty = -0.05 * (hull_angular_vel ** 2)
    balance_penalty = angle_penalty + angular_vel_penalty

    # ============================================================
    # 4. both_off_ground_penalty: 保留原样（僵尸但无害）
    # ============================================================
    contact_sum = leg1_contact + leg2_contact
    both_off_ground_penalty = -0.2 * (1.0 - contact_sum) if contact_sum < 0.5 else 0.0

    total_reward = forward_progress + gait_bonus + balance_penalty + both_off_ground_penalty

    components = {
        "forward_progress": forward_progress,
        "gait_bonus": gait_bonus,
        "balance_penalty": balance_penalty,
        "both_off_ground_penalty": both_off_ground_penalty,
        "gait_quality": gait_quality
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: forward_progress 信号几乎死亡，gait_quality 主导且与前进解耦，balance_penalty 压制前倾探索——agent 原地完美踏步。
- **behavior**: 存活 1600 步不摔倒，但 horizontal_velocity≈0，通过完美的双足交替获取高 gait_bonus，形成局部最优 exploit。
- **signal**: 缺少允许前倾的探索自由；forward_progress 在零速区梯度为零，无法从零速启动。
- **level**: Level 2（把 balance_penalty 从全局二次改为 hinge，释放探索空间）。
- **hypothesis**: 移除对微小倾斜的惩罚后，agent 会尝试前倾，从而获得 forward velocity；forward_progress 的平方项将从零启动并自我放大，把 agent 拉出原地踏步陷阱。
- **risk**: hinge 松弛可能让 agent 在学习初期频繁探索不安全倾斜导致早期摔倒（len 下降），但随着 forward_progress 激活应逐渐恢复稳定性。
