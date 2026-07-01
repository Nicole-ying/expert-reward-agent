# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订专家。接收训练反馈、环境约束和历史记录，分析问题并修改奖励函数代码。

你可以使用以下工具：
- search_reward_design_knowledge(query)：搜索设计技法库，查找解决特定问题的方法。query 用自然语言描述症状。
- get_skeleton_detail(skeleton_name)：获取某个骨架的数学形态、原理、陷阱和推荐配合。

决策原则：
1. 优先从组件证据的 ratio_to_progress 识别最严重的问题。ratio_to_progress 绝对值 > 0.5 的惩罚项需要优先处理。
2. 每次修订尽量只修改一个组件（系数微调、添加或删除），最多涉及两个。避免大改动。
3. 只有当同一骨架在历史中已尝试 ≥3 次且毫无改善，才考虑更换一个核心骨架（rebuild）。
4. 如果 best_reward 得分明显高于 current_reward，优先考虑 revert 到 best 的系数配置再微调。
5. 禁止使用环境契约中明确禁止的信号（terminal_success_reward、terminal_failure_penalty、original_reward 等）。
6. 禁止发明未声明的 info 字段，禁止 import/eval/open。

输出要求：
- 先写简短注释说明诊断和修改理由，然后输出完整 Python 代码。
- 函数签名：def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
- 返回 (float(total_reward), components_dict)，components 包含所有组件和 total_reward。
- components dict 只放 total_reward 公式中直接出现的变量名，不放中间计算变量。

```

## User Prompt

```markdown
# 环境契约
- function_signature: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
- allowed: obs[0..7], next_obs[0..7], action (0=noop,1=left,2=main,3=right), info (no reliable fields)
- forbidden: original_reward, official_reward, terminal_success_reward, terminal_failure_penalty


# 当前奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 目标位置假设为 (0, 0)，因为 obs[0] 和 obs[1] 是相对于目标着陆平台的坐标
    # 计算当前距离平方和下一时刻距离平方
    current_dist_sq = obs[0] ** 2 + obs[1] ** 2
    next_dist_sq = next_obs[0] ** 2 + next_obs[1] ** 2
    
    # progress_delta: 正数表示更接近目标
    progress_delta = current_dist_sq - next_dist_sq
    
    # 缩放因子，使奖励值在合理范围
    progress_scale = 2.0
    progress_delta_reward = progress_scale * progress_delta
    
    # ========== 稳定/安全约束：stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度
    # 使用 next_obs 因为动作效果体现在下一状态
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚：鼓励减速接近目标
    speed = (vel_x ** 2 + vel_y ** 2) ** 0.5
    speed_penalty_weight = 0.3
    speed_penalty = -speed_penalty_weight * speed
    
    # 姿态角惩罚：鼓励保持水平姿态（角度接近0）
    angle_penalty_weight = 0.2
    angle_penalty = -angle_penalty_weight * abs(body_angle)
    
    # 角速度惩罚：鼓励稳定姿态
    angular_vel_penalty_weight = 0.1
    angular_vel_penalty = -angular_vel_penalty_weight * abs(angular_vel)
    
    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty
    
    # ========== 任务完成 proxy：soft_landing_proxy ==========
    # 当飞行器接近目标、速度低、姿态稳定且双支撑接触时给予小奖励
    # 条件：距离近、速度低、姿态角小、双接触
    near_target_threshold = 0.5
    low_speed_threshold = 0.3
    stable_angle_threshold = 0.2
    
    is_near_target = current_dist_sq ** 0.5 < near_target_threshold
    is_low_speed = speed < low_speed_threshold
    is_stable_angle = abs(body_angle) < stable_angle_threshold
    is_both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    
    soft_landing_bonus = 0.0
    if is_near_target and is_low_speed and is_stable_angle and is_both_contact:
        soft_landing_bonus = 1.0
    
    soft_landing_proxy = soft_landing_bonus
    
    # ========== 组合总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy
    
    # ========== 构建 components dict ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# 训练反馈
# Training Feedback

## Training outcome
score=-111.554261, len=74.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.051732 | 0.055383 | 1.000000 | 1.000000 |
| soft_landing_proxy | 0.004579 | 0.004579 | 0.004579 | 0.088519 |
| stability_penalty | -0.330070 | 0.330070 | 1.000000 | -6.380333 |
| total_reward | -0.273758 | 0.282642 | 1.000000 | -5.291815 |
| generated_reward | -0.273758 | 0.282642 | 1.000000 | -5.291815 |
| original_env_reward | -1.513035 | 2.509036 | 1.000000 | -29.247325 |

## Distribution
- score: mean=-111.554261, min=-123.301028, max=-97.900804
- episode_length: mean=74.100000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -111.55 | -111.55 | 0.00 | 74.10 | progress_delta_reward=0.052 soft_landing_proxy=0.005 stability_penalty=-0.330 | new_best |

```
