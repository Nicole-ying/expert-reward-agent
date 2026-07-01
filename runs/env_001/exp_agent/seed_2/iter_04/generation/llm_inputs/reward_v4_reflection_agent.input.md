# ⚠️ 上一版代码验证失败
错误信息：Reward v4 failed validation: runs\env_001\exp_agent\seed_2\iter_04\generation\validations\reward_v4.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 当前奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 诊断 1：stability_penalty 的 ratio_to_progress = -0.089，已正常 ✅
    # 诊断 2：landing_bonus 的 nonzero_rate = 0.6%，仍然几乎不触发
    #   原因：contact_factor 是二进制门控，只要未接触地面乘积就为 0
    #   修复：去掉 contact_factor，只保留距离、速度、姿态角的连续乘积
    #   这样 agent 在靠近目标的过程中每一步都能感受到梯度
    # ============================================================
    
    # 主学习信号：progress_delta_reward（保持不变）
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta

    # 稳定约束：stability_penalty（系数已降低 10 倍，保持）
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.05 * abs(next_obs[4])
    angular_vel_penalty = 0.02 * abs(next_obs[5])
    speed_penalty = 0.01 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # ============================================================
    # 修复：去掉 contact_factor 的二进制门控
    # 连续乘积只基于距离、速度、姿态角，每个因子用 max(0, 1-x/threshold)
    # 这样 agent 在靠近目标、减速、摆正姿态时都能获得梯度
    # 系数从 5.0 提高到 10.0，因为去掉 contact 后信号更纯粹
    # ============================================================
    
    # 距离因子：距离 < 0.5 时为正，越近越大
    dist_factor = max(0.0, 1.0 - next_dist / 0.5)
    # 速度因子：速度 < 0.5 时为正，越慢越大
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    # 姿态角因子：角度 < 0.3 时为正，越小越大
    angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
    
    # 连续乘积：每个因子都在 [0,1]，乘积也在 [0,1]
    # 去掉 contact_factor，让梯度在每一步都能传递
    landing_bonus = 10.0 * dist_factor * speed_factor * angle_factor

    # 动作代价：energy_penalty（保持不变）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -0.05 * engine_use

    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward
    # 计算当前位置到目标（0,0）的距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta  # 正奖励表示更接近目标

    # 稳定约束：stability_penalty
    # 惩罚高速、大姿态角和大角速度
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_obs[4])  # 姿态角惩罚
    angular_vel_penalty = 0.2 * abs(next_obs[5])  # 角速度惩罚
    speed_penalty = 0.1 * speed  # 速度惩罚
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 任务完成 proxy：soft_landing_proxy
    # 当接近目标、低速、小姿态角且双接触时给予小奖励
    near_target = next_dist < 0.5
    low_speed = speed < 0.5
    stable_angle = abs(next_obs[4]) < 0.3
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 动作代价：energy_penalty（小权重）
    # 惩罚使用引擎的动作（action 1,2,3）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -0.05 * engine_use

    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty

    # 构建组件字典
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈
# Training Feedback

## Training outcome
score=-115.024416, len=71.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.005695 | 0.005695 | 0.113908 | -0.035362 |
| landing_bonus | 0.042785 | 0.042785 | 0.007405 | 0.265647 |
| progress_reward | 0.161058 | 0.170415 | 0.999997 | 1.000000 |
| stability_penalty | -0.014429 | 0.014429 | 1.000000 | -0.089590 |
| total_reward | 0.183718 | 0.199111 | 1.000000 | 1.140695 |
| generated_reward | 0.183718 | 0.199111 | 1.000000 | 1.140695 |
| original_env_reward | -1.617069 | 2.341746 | 1.000000 | -10.040299 |

## Distribution
- score: mean=-115.024416, min=-130.746612, max=-98.057243
- episode_length: mean=71.900000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -108.81 | -108.81 | 0.00 | 72.00 | energy_penalty=-0.007 landing_bonus=0.013 progress_reward=0.161 stability_penalty=-0.129 | new_best |
| 2 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -110.34 | -108.81 | -1.52 | 72.00 | energy_penalty=-0.006 landing_bonus=0.018 progress_reward=0.161 stability_penalty=-0.014 | no_meaningful_improvement |
| 3 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -115.02 | -108.81 | -6.21 | 71.90 | energy_penalty=-0.006 landing_bonus=0.043 progress_reward=0.161 stability_penalty=-0.014 | no_meaningful_improvement |
