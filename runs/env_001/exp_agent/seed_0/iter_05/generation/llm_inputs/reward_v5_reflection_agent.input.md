# ⚠️ 上一版代码验证失败
错误信息：Reward v5 failed validation: runs\env_001\exp_agent\seed_0\iter_05\generation\validations\reward_v5.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 261.780150）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 提取状态 ──
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # ── 距离 ──
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5

    # ── 速率标量 ──
    speed = (vx**2 + vy**2) ** 0.5

    # ═══════════════════════════════════════════
    # 主学习信号：进度增量奖励
    # ═══════════════════════════════════════════
    # 每一步向原点靠近就奖励，远离就惩罚
    progress_delta = dist - next_dist
    # mild clip 防目标附近震荡（skeleton 文档推荐）
    progress_delta = max(-0.5, min(0.5, progress_delta))

    # ═══════════════════════════════════════════
    # 软着陆 proxy：连续 3 因子乘积（无 contact）
    # ═══════════════════════════════════════════
    # 去掉 contact_factor —— 飞行中接触永远为 0，会让整个乘积归零
    # 去掉 ang_vel_factor —— 减少因子数，提升 nonzero_rate
    # 三个因子：靠近原点 + 低速 + 姿态正
    prox_factor = max(0.0, 1.0 - next_dist / 1.0)       # 距离 < 1.0 时有梯度
    vel_factor  = max(0.0, 1.0 - speed / 0.5)            # 速率 < 0.5 时有梯度
    ang_factor  = max(0.0, 1.0 - abs(angle) / 0.3)       # 倾角 < 0.3 时有梯度

    soft_landing_proxy = prox_factor * vel_factor * ang_factor

    # ═══════════════════════════════════════════
    # 轻量稳定性惩罚（大幅削弱，目标 < progress 的 10%）
    # ═══════════════════════════════════════════
    # 只保留角速度惩罚，抑制疯狂旋转
    stability_penalty = -0.005 * abs(ang_vel)

    # ═══════════════════════════════════════════
    # 总奖励
    # ═══════════════════════════════════════════
    w_progress = 5.0
    w_soft     = 1.0
    w_stab     = 1.0

    total_reward = (
        w_progress * progress_delta +
        w_soft     * soft_landing_proxy +
        w_stab     * stability_penalty
    )

    components = {
        "progress_delta_reward": progress_delta,
        "soft_landing_proxy": soft_landing_proxy,
        "stability_penalty": stability_penalty,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=261.780150, len=284.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.003556 | 0.003911 | 0.998337 | 1.000000 |
| soft_landing_proxy | 0.484643 | 0.484643 | 0.713254 | 136.297445 |
| stability_penalty | -0.000646 | 0.000646 | 0.999985 | -0.181692 |
| total_reward | 0.501775 | 0.502800 | 1.000000 | 141.115752 |
| generated_reward | 0.501775 | 0.502800 | 1.000000 | 141.115752 |
| original_env_reward | -0.143712 | 3.184503 | 1.000000 | -40.416600 |

## Distribution
- score: mean=261.780150, min=232.724231, max=291.172646
- episode_length: mean=284.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -107.10 | -107.10 | 0.00 | 74.20 | progress_delta_reward=0.016 soft_landing_proxy=0.005 stability_penalty=-0.142 | new_best |
| 2 | progress_delta_reward + soft_landing_proxy + stability_penalty | -111.45 | -107.10 | -4.35 | 74.10 | progress_delta_reward=0.016 soft_landing_proxy=0.003 stability_penalty=-0.058 | no_meaningful_improvement |
| 3 | angular_vel_penalty + energy_penalty + potential_shaping + soft_landing_proxy | -1771.11 | -107.10 | -1664.01 | 213.50 | angular_vel_penalty=-0.001 energy_penalty=-0.020 potential_shaping=0.130 soft_landing_proxy=0.001 | no_meaningful_improvement |
| 4 | progress_delta_reward + soft_landing_proxy + stability_penalty | 261.78 | 261.78 | 0.00 | 284.00 | progress_delta_reward=0.004 soft_landing_proxy=0.485 stability_penalty=-0.001 | target_solved_new_best |
