# 上一轮奖励函数代码（该轮得分: -113.309022）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 使用 next_obs 计算所有奖励组件，确保奖励反映动作后的状态
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. 主学习信号：密集距离奖励（负欧氏距离）
    distance = (x**2 + y**2) ** 0.5
    dist_reward = -distance

    # 2. 稳定约束：惩罚速度、姿态角和角速度
    stability_penalty = -0.1 * (abs(vx) + abs(vy)) \
                        - 0.2 * abs(angle) \
                        - 0.05 * abs(angular_vel)

    # 3. 任务完成近似信号：软着陆奖励（仅当多条件同时满足时激活）
    landing_proxy = 0.0
    if (abs(x) < 0.1 and abs(y) < 0.1 and 
        abs(vx) < 0.2 and abs(vy) < 0.2 and 
        abs(angle) < 0.1 and 
        left_contact == 1.0 and right_contact == 1.0):
        landing_proxy = 1.0

    total_reward = dist_reward + stability_penalty + landing_proxy

    components = {
        "dist_reward": dist_reward,
        "stability_penalty": stability_penalty,
        "landing_proxy": landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-113.309022, len=71.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| dist_reward | -0.972053 | 0.972053 | 1.000000 | -0.972053 |
| landing_proxy | 0.002430 | 0.002430 | 0.002430 | 0.002430 |
| stability_penalty | -0.144918 | 0.144918 | 1.000000 | -0.144918 |
| total_reward | -1.114541 | 1.119054 | 1.000000 | -1.114541 |
| generated_reward | -1.114541 | 1.119054 | 1.000000 | -1.114541 |
| original_env_reward | -1.642424 | 2.416951 | 1.000000 | -1.642424 |

## Distribution
- score: mean=-113.309022, min=-124.107854, max=-103.429116
- episode_length: mean=71.900000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress + soft_landing_proxy + stability_penalty | -105.90 | -105.90 | 0.00 | 72.00 | progress=0.016 soft_landing_proxy=0.002 stability_penalty=-0.014 | new_best |
| 2 | progress + soft_landing_proxy + stability_penalty | 187.93 | 187.93 | 0.00 | 694.50 | progress=0.003 soft_landing_proxy=0.257 stability_penalty=-0.001 | new_best |
| 3 | progress + soft_landing_continuous + stability_penalty | 143.84 | 187.93 | -44.08 | 1000.00 | progress=0.003 soft_landing_continuous=0.237 stability_penalty=-0.001 | no_meaningful_improvement |
| 4 | progress + soft_landing_continuous + stability_penalty | 137.07 | 187.93 | -50.85 | 921.60 | progress=0.003 soft_landing_continuous=0.042 stability_penalty=-0.001 | no_meaningful_improvement |
| 5 | progress + soft_landing_proxy + stability_penalty | 144.59 | 187.93 | -43.34 | 1000.00 | progress=0.003 soft_landing_proxy=0.244 stability_penalty=-0.001 | unsolved_stagnation_fresh_restart |
| 6 | dist_reward + landing_proxy + stability_penalty | -113.31 | -113.31 | 0.00 | 71.90 | dist_reward=-0.972 landing_proxy=0.002 stability_penalty=-0.145 | new_best |
