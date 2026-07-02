# ⚠️ 上一版代码验证失败
错误信息：Reward v2 failed validation: runs\env_001\exp_agent\seed_0\iter_02\generation\validations\reward_v2.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -107.097651）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取当前和下一时刻的位置
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    # 速度、姿态、接触信息（使用 next_obs 更合理，反映动作导致的后果）
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 主学习信号：进度增量奖励 ----------
    # 每一步奖励“当前到目标的距离”与“下一步到目标的距离”之差
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5
    progress_delta = dist - next_dist   # 正值表示更靠近目标

    # ---------- 稳定/安全惩罚 ----------
    # 惩罚水平、垂直速度，以及姿态角和角速度，鼓励稳定接近
    stability_penalty = -(
        0.1 * abs(vx) +
        0.1 * abs(vy) +
        0.2 * abs(angle) +
        0.1 * abs(ang_vel)
    )

    # ---------- 任务完成近似信号（软着陆 proxy） ----------
    # 同时满足：靠近中心、低速、姿态稳定、双支撑脚接触，则给予小奖励
    dist_thresh = 0.5
    vel_thresh = 0.2
    angle_thresh = 0.1
    ang_vel_thresh = 0.1

    if (next_dist < dist_thresh and
        abs(vx) < vel_thresh and
        abs(vy) < vel_thresh and
        abs(angle) < angle_thresh and
        abs(ang_vel) < ang_vel_thresh and
        left_contact > 0.5 and right_contact > 0.5):
        soft_landing_proxy = 1.0
    else:
        soft_landing_proxy = 0.0

    # ---------- 总奖励 ----------
    w_progress = 5.0
    w_stab = 1.0      # stability_penalty 内部已含负号，直接加
    w_soft = 1.0

    total_reward = (
        w_progress * progress_delta +
        w_stab * stability_penalty +
        w_soft * soft_landing_proxy
    )

    components = {
        "progress_delta_reward": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-107.097651, len=74.200000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.016108 | 0.017038 | 0.999992 | 1.000000 |
| soft_landing_proxy | 0.004557 | 0.004557 | 0.004557 | 0.282897 |
| stability_penalty | -0.141834 | 0.141834 | 1.000000 | -8.805068 |
| total_reward | -0.056736 | 0.066443 | 1.000000 | -3.522171 |
| generated_reward | -0.056736 | 0.066443 | 1.000000 | -3.522171 |
| original_env_reward | -1.538305 | 2.393403 | 1.000000 | -95.498350 |

## Distribution
- score: mean=-107.097651, min=-114.859921, max=-87.996822
- episode_length: mean=74.200000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -107.10 | -107.10 | 0.00 | 74.20 | progress_delta_reward=0.016 soft_landing_proxy=0.005 stability_penalty=-0.142 | new_best |
| 2 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -114.99 | -115.52 | 0.53 | 74.10 | energy_penalty=-0.001 progress_delta_reward=0.016 soft_landing_proxy=0.009 stability_penalty=-0.007 | no_meaningful_improvement |
| 3 | angular_vel_penalty + energy_penalty + potential_shaping + soft_landing_proxy | 242.09 | 242.09 | 0.00 | 441.30 | angular_vel_penalty=-0.001 energy_penalty=-0.008 potential_shaping=0.021 soft_landing_proxy=0.829 | target_solved_new_best |
| 4 | angular_vel_penalty + energy_penalty + potential_shaping + soft_landing_proxy | 179.44 | 242.09 | -62.64 | 812.60 | angular_vel_penalty=-0.001 energy_penalty=-0.007 potential_shaping=0.045 soft_landing_proxy=0.336 | stop_after_solved_drop_keep_best |
