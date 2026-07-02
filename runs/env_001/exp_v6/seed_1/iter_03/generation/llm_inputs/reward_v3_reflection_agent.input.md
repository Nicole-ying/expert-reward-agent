# 上一轮奖励函数代码（该轮得分: 218.095810）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 配置参数 ---
    # 主学习信号权重
    w_progress = 1.0

    # 稳定性惩罚 — 大幅削弱（上一轮 ratio 0.68，压制了 progress）
    w_speed = 0.001    # 原 0.01 → 降 10x
    w_angle = 0.001    # 原 0.01 → 降 10x
    w_angvel = 0.0001  # 原 0.001 → 降 10x

    # 软着陆代理 — 从二值改为连续乘积，提供密集梯度
    w_proxy = 0.2
    dist_threshold = 0.5       # 放宽距离阈值，配合连续形式
    speed_threshold = 0.3      # 放宽速度阈值
    angle_threshold = 0.5      # 放宽角度阈值

    # --- 1. 进度差分奖励 (主学习信号) ---
    d_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = d_current - d_next

    # --- 2. 稳定性惩罚 (大幅削弱，避免压制 progress) ---
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angvel_abs = abs(next_obs[5])

    stability_penalty = -(w_speed * speed + w_angle * angle_abs + w_angvel * angvel_abs)

    # --- 3. 软着陆代理 (连续乘积 → 密集梯度，非零率从 0.38% 提升) ---
    # 每个因子用 bounded max(0, 1 - x/D) 形式，取值范围 [0, 1]，自动限制
    near_factor = max(0.0, 1.0 - d_next / dist_threshold)
    speed_factor = max(0.0, 1.0 - speed / speed_threshold)
    angle_factor = max(0.0, 1.0 - angle_abs / angle_threshold)
    # 腿部接触：用平均接触量替代二值判断，提供部分梯度（单腿触地也有信号）
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0

    soft_proxy = w_proxy * near_factor * speed_factor * angle_factor * contact_factor

    # --- 总奖励 ---
    total_reward = progress + stability_penalty + soft_proxy

    components = {
        'progress': progress,
        'stability_penalty': stability_penalty,
        'soft_proxy': soft_proxy,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=218.095810, len=857.800000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress | 0.003960 | 0.004198 | 0.999233 | 0.003960 |
| soft_proxy | 0.105971 | 0.105971 | 0.617093 | 0.105971 |
| stability_penalty | -0.000352 | 0.000352 | 1.000000 | -0.000352 |
| total_reward | 0.109579 | 0.109785 | 1.000000 | 0.109579 |
| generated_reward | 0.109579 | 0.109785 | 1.000000 | 0.109579 |
| original_env_reward | -0.097533 | 1.448498 | 1.000000 | -0.097533 |

## Distribution
- score: mean=218.095810, min=169.284658, max=279.820484
- episode_length: mean=857.800000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress + soft_proxy + stability_penalty | -110.24 | -110.24 | 0.00 | 73.60 | progress=0.016 soft_proxy=0.001 stability_penalty=-0.011 | new_best |
| 2 | progress + soft_proxy + stability_penalty | 218.10 | 218.10 | 0.00 | 857.80 | progress=0.004 soft_proxy=0.106 stability_penalty=-0.000 | target_solved_new_best |
