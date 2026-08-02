# Response Record

```markdown
# 设计理由
**审计结论**：观测维度已全部使用，但 `landing_quality` 的连续乘积形式允许 agent 在平台附近悬停、单腿接触甚至仅位置/姿态良好即可持续获赏，导致 proxy exploit（episode 大幅延长、得分崩溃）。不存在未使用的关键观测，问题在于信号校准——代理信号与任务完成的“成功终止”严重脱节。

**行为**：agent 正在利用 `landing_quality` 的宽松连续信号，在目标平台附近徘徊、维持高 `landing_quality` 累积（episode_sum_mean=1018），避免触发任何终止（truncated=19/20），从而最大化 episode 总奖励，完全放弃了实际着陆与任务完成。

**信号**：`landing_quality` 作为软着陆品质信号，未与“双腿同时接触且稳定停靠”这个成功必要条件硬绑定，导致 agent 可以通过不产生完整双腿接触的姿态获得可观正向奖励。`landing_velocity_penalty` 因只在双腿接触时激活（active_rate=0.4%）而被绕过。

**层级**：Level 2 — 结构变换。对 `landing_quality` 添加双腿接触门控，使其仅当 `nleft_contact > 0.5 and nright_contact > 0.5` 时才激活，切断悬停/单腿接触的奖励流；同时适度降低系数，防止成功瞬时奖励过载。

**假设**：门控将 `landing_quality` 与真正着陆的必经步骤绑定，agent 失去徘徊的净收益后，会在 `progress`（仍有 20.0 系数）引导下接近平台，然后尝试双腿接触以获取着陆品质奖励，最终触发稳定着陆终止，`len` 应回落至接近 iter3 的 300~400，`score` 回归正值并可能超过 best 170。

**风险**：如果双腿接触状态难以在初期探索中出现，`landing_quality` 可能长时间为零，导致学习变慢；但 `progress` 的强引导（20.0）仍能提供足够的接近引力，总体风险可控。另外，agent 可能发展出“反复轻触双腿再抬离”的局部 exploitive 循环，但环境在双腿同时接触且速度较小时很可能触发 `body_not_awake_or_settled` 终止，抑制该行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆分
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft_contact = next_obs[6]
    nright_contact = next_obs[7]

    w_progress = 20.0
    w_landing = 0.5            # 降低系数，防止瞬时奖励过载
    w_land_vel = 10.0
    w_angle = 0.5
    w_angvel = 0.5
    engine_cost = 0.02

    dist = (x**2 + y**2) ** 0.5
    ndist = (nx**2 + ny**2) ** 0.5

    # 1. 距离改进（保持不变）
    progress = w_progress * (dist - ndist)

    # 2. 着陆质量软信号 —— 仅在双腿同时接触时激活
    if nleft_contact > 0.5 and nright_contact > 0.5:
        altitude_factor = max(0.0, 1.0 - abs(ny) / 0.5)
        align_factor    = max(0.0, 1.0 - abs(nx) / 0.5)
        vx_factor       = max(0.0, 1.0 - abs(nvx) / 0.3)
        vy_factor       = max(0.0, 1.0 - abs(nvy) / 0.5)
        angle_factor    = max(0.0, 1.0 - abs(nangle) / 0.2)
        product = (altitude_factor * align_factor * vx_factor *
                   vy_factor * angle_factor)
        if product > 0.0:
            landing_quality = w_landing * (product ** (1.0 / 5.0))
        else:
            landing_quality = 0.0
    else:
        landing_quality = 0.0

    # 3. 着陆速度惩罚（仅在双腿接触时）
    fcontact = float(nleft_contact * nright_contact)
    if fcontact > 0.5:
        vel_pen = -w_land_vel * (nvx**2 + nvy**2)
    else:
        vel_pen = 0.0

    # 4. 姿态稳定惩罚（全程）
    att_penalty = -w_angle * (nangle**2) - w_angvel * (nangvel**2)

    # 5. 引擎使用惩罚
    eng_pen = -engine_cost if action != 0 else 0.0

    total_reward = progress + landing_quality + vel_pen + att_penalty + eng_pen
    components = {
        "progress": progress,
        "landing_quality": landing_quality,
        "landing_velocity_penalty": vel_pen,
        "attitude_penalty": att_penalty,
        "engine_cost": eng_pen
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 全部观测已使用，`landing_quality` 与成功终止脱节导致 proxy exploit；加入双腿接触门控切断悬停奖励。
- **behavior**: agent 在平台附近徘徊/悬停以持续获得 `landing_quality`，避免任何终止，episode 几乎全部超时。
- **signal**: `landing_quality` 过于泛化，缺少对“双腿同时接触”的硬性要求，agent 学会规避真正的着陆。
- **level**: Level 2（结构变换：添加门控，降低系数）
- **hypothesis**: 门控使徘徊的净收益消失，agent 被迫完成双腿着陆才能获得正向信号，episode 长度将缩短，得分回升。
- **risk**: 门控可能增加早期探索难度，但 `progress` 的强引力可缓解；仍存在轻触双腿后立即离地的局部 exploit 可能，但环境终止机制预计会抑制。
