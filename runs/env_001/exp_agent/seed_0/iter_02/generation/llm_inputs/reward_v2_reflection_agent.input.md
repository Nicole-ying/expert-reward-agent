# 环境契约
- function_signature: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
- allowed: obs[0..7], next_obs[0..7], action (0=noop,1=left,2=main,3=right), info (no reliable fields)
- forbidden: original_reward, official_reward, terminal_success_reward, terminal_failure_penalty


# 当前奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. 主学习信号: progress_delta_reward
    #    基于当前位置到目标(0,0)的距离变化，引导飞行器接近目标
    # ============================================================
    # 当前距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    # 下一步距离
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    # 距离减少为正奖励，增加为负奖励
    progress_delta = current_dist - next_dist
    # 缩放因子，使奖励值在合理范围
    progress_scale = 2.0
    progress_delta_reward = progress_delta * progress_scale

    # ============================================================
    # 2. 稳定/安全约束: stability_penalty
    #    惩罚高速、大姿态角和大角速度，鼓励稳定接近
    # ============================================================
    # 速度惩罚（使用next_obs，因为动作执行后的状态）
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_penalty_weight = 0.1
    speed_penalty = -speed_penalty_weight * speed

    # 姿态角惩罚（角度偏离0度）
    angle_penalty_weight = 0.05
    angle_penalty = -angle_penalty_weight * abs(next_obs[4])

    # 角速度惩罚
    angular_vel_penalty_weight = 0.02
    angular_vel_penalty = -angular_vel_penalty_weight * abs(next_obs[5])

    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty

    # ============================================================
    # 3. 任务完成proxy: soft_landing_proxy
    #    当飞行器接近目标、速度低、姿态稳定且双支撑接触时给予小奖励
    # ============================================================
    # 条件：距离目标很近（<0.5）
    near_target = 1.0 if next_dist < 0.5 else 0.0
    # 条件：速度很低（<0.3）
    low_speed = 1.0 if speed < 0.3 else 0.0
    # 条件：姿态角很小（<0.2弧度）
    stable_angle = 1.0 if abs(next_obs[4]) < 0.2 else 0.0
    # 条件：双支撑接触
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    # 所有条件满足时给予小奖励
    landing_bonus_weight = 1.0
    soft_landing_proxy = landing_bonus_weight * near_target * low_speed * stable_angle * both_contact

    # ============================================================
    # 4. 动作代价: energy_penalty（小权重）
    #    轻微惩罚使用引擎，鼓励节能
    # ============================================================
    # action 0: no_engine -> 无惩罚
    # action 1,2,3: 使用引擎 -> 小惩罚
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * engine_use

    # ============================================================
    # 总奖励
    # ============================================================
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy + energy_penalty

    # 组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈
# Training Feedback

## Training outcome
score=-115.524472, len=74.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.001737 | 0.001737 | 0.173727 | -0.053713 |
| progress_delta_reward | 0.032344 | 0.034174 | 0.999996 | 1.000000 |
| soft_landing_proxy | 0.005204 | 0.005204 | 0.005204 | 0.160886 |
| stability_penalty | -0.108909 | 0.108909 | 1.000000 | -3.367223 |
| total_reward | -0.073099 | 0.083249 | 1.000000 | -2.260049 |
| generated_reward | -0.073099 | 0.083249 | 1.000000 | -2.260049 |
| original_env_reward | -1.581795 | 2.341473 | 1.000000 | -48.905611 |

## Distribution
- score: mean=-115.524472, min=-130.746612, max=-105.791517
- episode_length: mean=74.100000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_proxy + stability_penalty | -115.52 | -115.52 | 0.00 | 74.10 | energy_penalty=-0.002 progress_delta_reward=0.032 soft_landing_proxy=0.005 stability_penalty=-0.109 | new_best |
