# Response Record

# 设计理由
**修改组件**：`landing_bonus`（着陆引导奖励）。  
**当前问题**：该奖励在脚部接触时激活，但速度、角度、距离的阈值过于宽松（速度阈值 1.0、角度阈值 0.5、距离衰减常数 0.8），导致 agent 在接触但未稳定（有一定速度或倾角）时已能获得大量奖励（mean episode=652）。训练结果 agent 学会反复接触‑轻微弹跳‑再接触以持续获取奖励，而不是完成稳定着陆，因此 truncated 率达 18/20，episode 长度 933 步，距离成功只有微弱进步。  
**改变内容**：将 `landing_bonus` 改造为更严格的“稳定着陆”奖励。各因子改用线性缩短型（max(0, 1‑value/threshold)），并收紧阈值：距离阈值 0.5、总速度阈值 0.3、角度阈值 0.15。只有同时满足这些条件**且**腿部接触时，才能获得非零奖励，稳定时奖励最大。同时适度提高权重（1.0→2.0），以使稳定着陆时的每步奖励仍有竞争力（原平稳接触约 0.95/步，现稳定时 2.0/步）。  
**预期**：agent 被迫追求更低速度、更小倾角的状态才能获得高 reward，接触后自然会减速归零、正姿，从而触发环境内置的“稳定→终止”逻辑，提高成功着陆率。由于成功终止后 episode 不会无限累积奖励，最终平均回报会因成功率提高而接近或超过当前水平（目标 200）。  
**校准**：无新增惩罚负担；新奖励系数产生的 per‑step ≤ 2.0，为当前主信号（landing_bonus 均值 0.7/步）的 ~3 倍，但仅在稳定着陆短暂阶段出现，整体 episode 总奖励有望维持合理尺度。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, l_contact, r_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nl_contact, nr_contact = next_obs

    # 1. Main progress signal: distance reduction to target pad
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_obs - dist_next

    # 2. Stability constraints (quadratic penalties on next state)
    #    (kept as in previous version, though they had negligible effect)
    angle_penalty = -0.1 * (nangle ** 2)
    angvel_penalty = -0.05 * (nangvel ** 2)

    # 3. Contact-gated soft landing attractor (tightened conditions)
    #    Only active when support legs are in contact, and rewards are given
    #    only when proximity, speed, and angle are all within narrow bands.
    proximity = max(0.0, 1.0 - dist_next / 0.5)          # dist < 0.5
    speed = abs(nvx) + abs(nvy)
    speed_factor = max(0.0, 1.0 - speed / 0.3)          # speed < 0.3
    angle_factor = max(0.0, 1.0 - abs(nangle) / 0.15)   # |angle| < 0.15

    contact_gate = float(nl_contact or nr_contact)
    landing_attractor = proximity * speed_factor * angle_factor * contact_gate

    w_progress = 10.0
    w_attractor = 2.0    # slightly increased to compensate for tighter gating

    total = (w_progress * progress +
             angle_penalty + angvel_penalty +
             w_attractor * landing_attractor)

    components = {
        "progress": w_progress * progress,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_bonus": w_attractor * landing_attractor
    }

    return float(total), components
```

# 诊断摘要
- **audit**: 信号齐全但校准偏差——`landing_bonus` 的阈值过宽，使不稳定接触也能获得高价奖励，导致 agent 学会“接触徘徊”而非“稳定着陆”。
- **behavior**: agent 成功触地并持续获得 `landing_bonus`，但未减速归零、未调整姿态，因此 episode 长时间徘徊（len≈933），truncated 占 90%，极少真正成功。
- **signal**: 缺少对稳定着陆（近零速度、小倾角、极近距离）的严格奖励梯度；现有 `landing_bonus` 在 73% 活跃率下掩盖了这一需求。
- **level**: Level 2 — 结构变换（收紧组件阈值，将宽松吸引子改为严格稳定着陆条件）。
- **hypothesis**: 收紧条件后，agent 只有同时满足接触、近距、低速、小倾角才能获得正向奖励，这会将优化方向从“频繁接触”转为“稳定着陆”，从而提高成功终止率并可能提升总得分。
- **risk**: 阈值过于严格可能使早期探索难以获得 `landing_bonus`，若 progress 信号不足，可能导致学习停滞或退步；若稳定着陆时间过短，episode 总奖励可能下降。
